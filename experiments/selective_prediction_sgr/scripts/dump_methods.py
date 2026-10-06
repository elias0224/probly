"""Dump predictions and criteria of the post-training methods on the same datasets as ``dump_shift.py``.

Writes ``runs/seed{S}/shift/{method}/{dataset}.npz`` (labels, mean probabilities and the method's criteria) for the
methods finetune, swag, laplace, gda, ddu, vbll and sngp (plus the SNGP variants sngp_long and sngp_scratch and dropout_scratch, MC dropout trained from scratch with
the deterministic eval-mode softmax ``softmax_det`` next to the MC criteria); finished files
are skipped. SWAG, DDU, VBLL, SNGP and finetune need the weights from ``train.py``; Laplace and the density heads of GDA and DDU are cheap and refit on the training set at the
start of every run that has something left to dump.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
import time

from dump_shift import ALL_DATASETS, load_dataset
import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from probly.layers.torch import GaussianMixtureHead
from probly.method.ddu import negative_log_density
from probly.predictor import predict as probly_predict
from probly.quantification import quantify
from probly.representer import representer
from sgr_experiment.data import load_cifar10, normalize
from sgr_experiment.loaders import load_base, load_ddu, load_dropout, load_finetune, load_sngp, load_swag, load_vbll
from sgr_experiment.model import disable_dropout
from sgr_experiment.uncertainty import summarize_samples, torch_one_minus_max
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, run_dir, seed_everything

METHODS = ["finetune", "swag", "laplace", "gda", "ddu", "vbll", "sngp", "sngp_long", "sngp_scratch", "dropout_scratch"]
NEEDS = {
    "finetune": "finetune.pt",
    "swag": "swag.pt",
    "laplace": "base.pt",
    "gda": "base.pt",
    "ddu": "ddu.pt",
    "vbll": "vbll.pt",
    "sngp": "sngp.pt",
    "sngp_long": "sngp_long.pt",
    "sngp_scratch": "sngp_scratch.pt",
    "dropout_scratch": "dropout_scratch.pt",
}
Predict = Callable[[torch.Tensor], dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--subset", type=int, default=None, help="Use only the first N images of each dataset (smoke tests).")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=100, help="Samples of Laplace, VBLL and dropout_scratch.")
    p.add_argument("--swag-samples", type=int, default=30)
    p.add_argument("--swag-max-rank", type=int, default=20)
    p.add_argument("--swag-scale", type=float, default=0.5)
    p.add_argument(
        "--swag-bn-update",
        choices=["none", "per_sample"],
        default="none",
        help="per_sample recomputes the BatchNorm statistics on a fixed training subset for every drawn weight sample.",
    )
    p.add_argument("--swag-bn-samples", type=int, default=5000)
    p.add_argument("--p", type=float, default=0.5, help="Dropout probability of dropout_scratch (must match training).")
    p.add_argument("--sn-coeff", type=float, default=3.0)
    p.add_argument("--vbll-parameterization", default="dense")
    p.add_argument("--sngp-norm-multiplier", type=float, default=6.0)
    p.add_argument("--sngp-init-std", type=float, default=0.05)
    p.add_argument("--sngp-momentum", type=float, default=-1.0)
    return p.parse_args()


def softmax_dict(logits: torch.Tensor) -> dict[str, np.ndarray]:
    """``mean_probs`` and ``maxprob`` of a deterministic classifier."""
    p = torch.softmax(logits.float(), dim=-1)
    return {"mean_probs": p.cpu().numpy(), "maxprob": torch_one_minus_max(p).float().cpu().numpy()}


def update_bn(model: nn.Module, x_bn: torch.Tensor, batch_size: int = 1000) -> None:
    """Recompute the BatchNorm running statistics of ``model`` as the average over the normalized images ``x_bn``."""
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
        m.train()
    with torch.no_grad():
        for start in range(0, len(x_bn), batch_size):
            model(x_bn[start : start + batch_size])
    for m, mom in zip(bns, momenta, strict=True):
        m.momentum = mom
        m.eval()


def fit_features(extract: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, y: torch.Tensor, head: GaussianMixtureHead) -> None:
    """Fit the density head on the CPU on the features of all (un-augmented, normalized) training images."""
    feats = []
    with torch.no_grad():
        for start in range(0, len(x), 1000):
            feats.append(extract(x[start : start + 1000]).float().cpu())
    head.cpu().fit(torch.cat(feats), y.cpu())


def build_finetune(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:  # noqa: ARG001
    """Plain fine-tuned model, softmax criterion."""
    model = load_finetune(run_dir(args.runs, seed) / "finetune.pt").to(device)
    return lambda x: softmax_dict(model(x))


def build_swag(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:
    """SWAG via the probly representer, plus the deterministic SWA-mean softmax."""
    model = load_swag(run_dir(args.runs, seed) / "swag.pt", args.swag_max_rank, args.swag_scale).to(device)
    disable_dropout(model)  # the sampler would otherwise force the conv-block dropout of the base into train mode
    swa = disable_dropout(load_base(run_dir(args.runs, seed) / "base.pt")).to(device)  # same architecture as the wrapped model
    swa.load_state_dict(model.model.state_dict())
    vector_to_parameters(model.mean, swa.parameters())
    if args.swag_bn_update == "per_sample":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train[1]), generator=g)[: args.swag_bn_samples].to(train[0].device)
        x_bn = train[0][idx]
        update_bn(swa, x_bn)
        busy = [False]

        def hook(_module: nn.Module, _inputs: tuple) -> None:
            # Runs after the sampled weights are loaded into the wrapped model and before its forward pass.
            if busy[0]:
                return
            busy[0] = True
            try:
                update_bn(model.model, x_bn)
            finally:
                busy[0] = False

        model.model.register_forward_pre_hook(hook)
    rep = representer(model, num_samples=args.swag_samples)
    model.eval()
    swa.eval()

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        out = summarize_samples(rep.represent(x))
        out["softmax_swa"] = torch.softmax(swa(x).float(), dim=-1).cpu().numpy()
        return out

    return predict


def build_laplace(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:
    """Last-layer KFAC Laplace of the base model, fitted on the training set, with optimized prior precision."""
    base = load_base(run_dir(args.runs, seed) / "base.pt").to(device)
    la = Laplace(base, "classification", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(DataLoader(TensorDataset(*train), batch_size=500))
    la.optimize_prior_precision(method="marglik")
    rep = representer(la, num_samples=args.num_samples)
    return lambda x: summarize_samples(rep.represent(x))


def build_gda(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:
    """Gaussian discriminant analysis on the 512-dim features in front of the last Linear layer of the base model."""
    layers = list(load_base(run_dir(args.runs, seed) / "base.pt").to(device).children())
    extract, last = nn.Sequential(*layers[:-1]), layers[-1]
    head = GaussianMixtureHead(10, 512)
    fit_features(extract, *train, head)

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        feats = extract(x)
        out = softmax_dict(last(feats))
        out["density"] = negative_log_density(head(feats.float().cpu())).numpy().astype(np.float32)
        return out

    return predict


def build_ddu(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:
    """DDU: classifier softmax and the negative log density of the encoder features (probly representer)."""
    model = load_ddu(run_dir(args.runs, seed) / "ddu.pt", args.sn_coeff).to(device)
    fit_features(model.encoder, *train, model.density_head)
    model.density_head.to(device)
    rep = representer(model)

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        r = rep.represent(x)
        out = softmax_dict(torch.log(r.softmax.probabilities))
        out["density"] = quantify(r).epistemic.detach().cpu().numpy().astype(np.float32)
        return out

    return predict


def build_vbll(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:  # noqa: ARG001
    """VBLL: MC softmax over logits sampled from the closed-form predictive Gaussian (probly representer)."""
    model = load_vbll(run_dir(args.runs, seed) / "vbll.pt", args.vbll_parameterization).to(device)
    rep = representer(model, num_samples=args.num_samples)
    return lambda x: summarize_samples(rep.represent(x))


def build_sngp(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    train: tuple,  # noqa: ARG001
    checkpoint: str = "sngp.pt",
) -> Predict:
    """SNGP: softmax of the GP mean logits and the Dempster-Shafer epistemic score (probly decomposition)."""
    model = load_sngp(
        run_dir(args.runs, seed) / checkpoint,
        args.sngp_norm_multiplier,
        args.sngp_init_std,
        args.sngp_momentum,
    ).to(device)

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        logits, _ = model(x)
        out = softmax_dict(logits)
        out["ds"] = quantify(probly_predict(model, x)).epistemic.detach().cpu().numpy().astype(np.float32)
        return out

    return predict


def build_dropout_scratch(seed: int, args: argparse.Namespace, device: torch.device, train: tuple) -> Predict:  # noqa: ARG001
    """MC dropout trained from scratch (probly representer), plus the deterministic eval-mode softmax ``softmax_det`` (``sr``)."""
    model = load_dropout(run_dir(args.runs, seed) / "dropout_scratch.pt", p=args.p).to(device)
    rep = representer(model, num_samples=args.num_samples)

    def predict(x: torch.Tensor) -> dict[str, np.ndarray]:
        out = summarize_samples(rep.represent(x))
        model.eval()  # the sampler restores eval mode afterwards; make sure the deterministic pass has no dropout
        out["softmax_det"] = torch.softmax(model(x).float(), dim=-1).cpu().numpy()
        return out

    return predict


BUILDERS = {
    "finetune": build_finetune,
    "swag": build_swag,
    "laplace": build_laplace,
    "gda": build_gda,
    "ddu": build_ddu,
    "vbll": build_vbll,
    "sngp": build_sngp,
    "sngp_long": partial(build_sngp, checkpoint="sngp_long.pt"),
    "sngp_scratch": partial(build_sngp, checkpoint="sngp_scratch.pt"),
    "dropout_scratch": build_dropout_scratch,
}


def main() -> None:
    """Write ``runs/seed{S}/shift/{method}/{dataset}.npz`` for every missing seed x method x dataset triple."""
    args = parse_args()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    x_clean, y_clean = load_cifar10(args.data_dir, train=False)
    if args.subset:
        x_clean, y_clean = x_clean[: args.subset], y_clean[: args.subset]
    train: tuple[torch.Tensor, torch.Tensor] | None = None
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for seed in args.seeds:
        for method in args.methods:
            out_dir = run_dir(args.runs, seed) / "shift" / method
            todo = [d for d in args.datasets if not (out_dir / f"{d}.npz").exists()]
            if not todo:
                print(f"seed {seed} {method}: all requested datasets exist, skipping.")
                continue
            if not (run_dir(args.runs, seed) / NEEDS[method]).exists():
                print(f"seed {seed} {method}: {NEEDS[method]} not found, skipping.")
                continue
            if train is None:  # normalized, un-augmented training set (the subset only shrinks it for smoke tests)
                x_tr, y_tr = load_cifar10(args.data_dir, train=True, device=device)
                n_tr = args.subset or len(y_tr)
                train = (normalize(x_tr[:n_tr]), y_tr[:n_tr])
            seed_everything(seed + 98765)
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            predict = BUILDERS[method](seed, args, device, train)
            print(f"seed {seed} {method}: ready in {time.perf_counter() - t0:.0f}s", flush=True)
            for name in todo:
                if name not in cache:
                    x_u8, labels = load_dataset(name, args.data_dir, (x_clean, y_clean))
                    if args.subset:
                        x_u8, labels = x_u8[: args.subset], labels[: args.subset]
                    cache[name] = (x_u8, labels)
                x_u8, labels = cache[name]
                t0 = time.perf_counter()
                parts: dict[str, list[np.ndarray]] = {}
                for start in range(0, len(x_u8), args.batch_size):
                    x = normalize(x_u8[start : start + args.batch_size].to(device))
                    with torch.no_grad():
                        for k, v in predict(x).items():
                            parts.setdefault(k, []).append(v)
                res = {k: np.concatenate(v).astype(np.float32) for k, v in parts.items()}
                np.savez(out_dir / f"{name}.npz", labels=labels.numpy().astype(np.int64), **res)
                msg = f"seed {seed} {method} {name}: {len(labels)} images in {time.perf_counter() - t0:.0f}s"
                if name == "cifar10":
                    msg += f", acc {(res['mean_probs'].argmax(1) == labels.numpy()).mean():.4f}"
                print(msg, flush=True)


if __name__ == "__main__":
    main()
