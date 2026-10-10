"""ImageNet part (Sec. 5.3, Tables 3-6) of Geifman and El-Yaniv (2017), arXiv 1705.08500.

The paper takes pretrained VGG-16 and ResNet-50 models, splits the ILSVRC validation set (50k images) into two halves
of about 25k, runs SGR with softmax response (SR) on one half and reports risk and coverage on the other, for the
top-1 and the top-5 loss. This module holds the paper's numbers, the data and model loaders, the top-k losses and
criteria, and the split protocol shared by ``scripts/dump_imagenet.py`` and ``scripts/imagenet_table.py``.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import TYPE_CHECKING
import warnings

import numpy as np
import torch
from torchvision import datasets, models

from probly.metrics.selective_prediction import coverage_at_risk, risk_at_coverage
from probly.selective_prediction import SGRSelector
from sgr_experiment.metrics import apply_threshold, risk_bound, threshold_for_risk

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import nn
    from torch.utils.data import Dataset

# Rows of Tables 3-6: r*, train risk, train coverage, test risk, test coverage, bound b*. delta = 0.001 throughout.
PAPER_TABLES: dict[tuple[str, str], list[tuple[float, float, float, float, float, float]]] = {
    ("vgg16", "top1"): [  # Table 3
        (0.02, 0.0161, 0.2355, 0.0131, 0.2322, 0.0200),
        (0.05, 0.0462, 0.4292, 0.0446, 0.4276, 0.0500),
        (0.10, 0.0964, 0.5968, 0.0948, 0.5951, 0.1000),
        (0.15, 0.1466, 0.7164, 0.1467, 0.7138, 0.1500),
        (0.20, 0.1937, 0.8131, 0.1949, 0.8154, 0.2000),
        (0.25, 0.2441, 0.9117, 0.2445, 0.9120, 0.2500),
    ],
    ("vgg16", "top5"): [  # Table 4
        (0.01, 0.0080, 0.3391, 0.0078, 0.3341, 0.0100),
        (0.02, 0.0181, 0.5360, 0.0179, 0.5351, 0.0200),
        (0.03, 0.0281, 0.6768, 0.0290, 0.6735, 0.0300),
        (0.04, 0.0381, 0.7610, 0.0379, 0.7586, 0.0400),
        (0.05, 0.0481, 0.8263, 0.0496, 0.8262, 0.0500),
        (0.06, 0.0563, 0.8654, 0.0577, 0.8668, 0.0600),
        (0.07, 0.0663, 0.9093, 0.0694, 0.9114, 0.0700),
    ],
    ("resnet50", "top1"): [  # Table 5
        (0.02, 0.0161, 0.2613, 0.0164, 0.2585, 0.0199),
        (0.05, 0.0462, 0.4906, 0.0474, 0.4878, 0.0500),
        (0.10, 0.0965, 0.6544, 0.0988, 0.6502, 0.1000),
        (0.15, 0.1466, 0.7711, 0.1475, 0.7676, 0.1500),
        (0.20, 0.1937, 0.8688, 0.1955, 0.8677, 0.2000),
        (0.25, 0.2441, 0.9634, 0.2451, 0.9614, 0.2500),
    ],
    ("resnet50", "top5"): [  # Table 6
        (0.01, 0.0080, 0.3796, 0.0085, 0.3807, 0.0099),
        (0.02, 0.0181, 0.5938, 0.0189, 0.5935, 0.0200),
        (0.03, 0.0281, 0.7122, 0.0273, 0.7096, 0.0300),
        (0.04, 0.0381, 0.8180, 0.0358, 0.8158, 0.0400),
        (0.05, 0.0481, 0.8856, 0.0464, 0.8846, 0.0500),
        (0.06, 0.0581, 0.9256, 0.0552, 0.9231, 0.0600),
        (0.07, 0.0663, 0.9508, 0.0629, 0.9484, 0.0700),
    ],
}
PAPER_TABLE_NUMBER = {("vgg16", "top1"): 3, ("vgg16", "top5"): 4, ("resnet50", "top1"): 5, ("resnet50", "top5"): 6}
PAPER_DELTA = 0.001
PAPER_M = 25000  # "approximately 25,000" per half
TASKS = {"top1": 1, "top5": 5}

# torchvision weights closest to 2017 (the V1 recipes) and their published single-crop accuracies (top-1, top-5).
MODELS: dict[str, tuple[Callable[..., nn.Module], models.WeightsEnum]] = {
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}
PUBLISHED_ACC = {"vgg16": (0.71592, 0.90382), "resnet50": (0.76130, 0.92862)}

# Criterion name -> (npz key of the criterion, npz key of the top-5 predictions, label). Lower = more confident.
CRITERIA = {
    "sr": ("sr", "top5", "SR, 1 - max softmax (paper)"),
    "top5_mass": ("top5_mass", "top5", "1 - top-5 softmax mass (loss-matched)"),
    "mc_variance": ("mc_variance", "mc_top5", "MC dropout, predicted-class variance"),
    "mc_maxprob": ("mc_maxprob", "mc_top5", "MC dropout, 1 - max mean prob"),
}


def load_model(name: str) -> tuple[nn.Module, Callable]:
    """Pretrained torchvision model in eval mode and its evaluation transform (resize 256, center crop 224)."""
    builder, weights = MODELS[name]
    return builder(weights=weights).eval(), weights.transforms()


def imagenet_val(root: Path, transform: Callable) -> Dataset:
    """The ILSVRC2012 validation set (50k images, labels in the usual sorted-wnid order).

    ``root/val`` with one folder per wnid is read as an image folder. Otherwise ``root`` must hold
    ``ILSVRC2012_img_val.tar`` and ``ILSVRC2012_devkit_t12.tar.gz``, which torchvision unpacks on first use.
    """
    val = root / "val"
    if val.is_dir() and sum(1 for d in val.iterdir() if d.is_dir()) == 1000:
        return datasets.ImageFolder(str(val), transform=transform)
    return datasets.ImageNet(str(root), split="val", transform=transform)


def torch_topk_complement(probs: torch.Tensor, k: int) -> torch.Tensor:
    """``1 - (sum of the k largest probabilities)`` as the float64 sum of the others, on the CPU, shape ``(n,)``.

    Summing the small probabilities keeps the ranking of confident predictions (see
    :func:`sgr_experiment.uncertainty.one_minus_max`, which is the case ``k = 1``).
    """
    return probs.detach().cpu().double().sort(-1).values[..., :-k].sum(-1)


def topk_losses(top5: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Zero-one top-``k`` loss: 1 if the label is not among the first ``k`` columns of ``top5``, float64 ``(n,)``."""
    return (~(np.asarray(top5)[:, :k] == np.asarray(labels)[:, None]).any(1)).astype(np.float64)


def sgr(criterion: np.ndarray, losses: np.ndarray, r: float, delta: float) -> tuple[float, float]:
    """Threshold and bound of probly's ``SGRSelector`` (-inf and NaN if nothing is certified, without the warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sel = SGRSelector(r, delta).calibrate(criterion, losses)
    return float(sel.threshold), float(sel.bound)


def empirical(criterion: np.ndarray, losses: np.ndarray, r: float, delta: float) -> tuple[float, float]:  # noqa: ARG001
    """Largest-coverage threshold whose selection-half risk is at most r* (no guarantee); bound is NaN."""
    return threshold_for_risk(criterion, losses, r), float("nan")


type Rule = Callable[[np.ndarray, np.ndarray, float, float], tuple[float, float]]
RULES: dict[str, Rule] = {"sgr": sgr, "emp": empirical}


def rules_with_sweep(deltas: list[float]) -> dict[str, Rule]:
    """:data:`RULES` plus ``sgr_delta_{d}``: probly's SGR at a fixed delta ``d`` instead of the run's delta.

    Shows which delta the paper's coverages correspond to (its printed bounds imply delta between about 0.01 and 0.1).
    """
    out = dict(RULES)
    for d in deltas:
        out[f"sgr_delta_{d:g}"] = lambda c, l, r, _delta, d=d: sgr(c, l, r, d)
    return out


def splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n_splits`` random halvings ``(selection, test)`` of ``range(n)``."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return [(p[:half], p[half:]) for p in (rng.permutation(n) for _ in range(n_splits))]


def evaluate_unit(
    criterion: np.ndarray,
    losses: np.ndarray,
    rows: list[tuple[float, ...]],
    halves: list[tuple[np.ndarray, np.ndarray]],
    delta: float,
    rules: dict[str, Rule] | None = None,
) -> dict[tuple, list[float]]:
    """Values per key over the splits, for one criterion and one loss.

    Keys are ``(rule, r*, column)`` with the columns ``certified``, ``sel_risk``, ``sel_cov``, ``test_risk``,
    ``test_cov``, ``bound`` and ``viol`` (test risk above r*), and ``(r*, "cov_at_paper")`` /
    ``(r*, "risk_at_paper")``: the test-half coverage at the paper's test risk and the test-half risk at the paper's
    test coverage. Risk, bound and violation values are NaN when the rule certified nothing.
    """
    rules = RULES if rules is None else rules
    acc: dict[tuple, list[float]] = defaultdict(list)
    for sel, test in halves:
        for r, _, _, p_risk, p_cov, _ in rows:
            acc[r, "cov_at_paper"].append(float(coverage_at_risk(criterion[test], losses[test], p_risk)))
            acc[r, "risk_at_paper"].append(float(risk_at_coverage(criterion[test], losses[test], p_cov)))
            for name, rule in rules.items():
                thr, bound = rule(criterion[sel], losses[sel], r, delta)
                certified = thr != -np.inf
                sel_risk, sel_cov = apply_threshold(criterion[sel], losses[sel], thr)
                test_risk, test_cov = apply_threshold(criterion[test], losses[test], thr)
                nan = float("nan")
                acc[name, r, "certified"].append(float(certified))
                acc[name, r, "sel_cov"].append(sel_cov)
                acc[name, r, "test_cov"].append(test_cov)
                acc[name, r, "sel_risk"].append(sel_risk if certified else nan)
                acc[name, r, "test_risk"].append(test_risk if certified else nan)
                acc[name, r, "bound"].append(bound if certified else nan)
                acc[name, r, "viol"].append(float(test_risk > r + 1e-12) if certified else nan)
    return acc


def paper_bound_check(model: str, task: str, m: int = PAPER_M, delta: float = PAPER_DELTA) -> list[dict[str, float]]:
    """Lemma 3.1 (Clopper-Pearson) bound that the paper's own train columns give, per row of Tables 3-6.

    The accepted count is ``round(train coverage * m)`` and the error count ``round(train risk * accepted)``; the
    bound is computed at ``delta / ceil(log2 m)`` as in Algorithm 1 and at the full ``delta``.
    """
    steps = math.ceil(math.log2(m))
    out = []
    for r, tr, tc, _, _, b in PAPER_TABLES[model, task]:
        n = round(tc * m)
        e = round(tr * n)
        out.append(
            {
                "r": r,
                "accepted": n,
                "errors": e,
                "paper_bound": b,
                "bound_split": risk_bound(e, n, delta / steps),
                "bound_full": risk_bound(e, n, delta),
            }
        )
    return out
