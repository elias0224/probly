"""Dump predictions and MC-dropout criteria on clean CIFAR-10, SVHN, CIFAR-100 and corrupted CIFAR-10 test sets.

Writes ``runs/seed{S}/shift/{dataset}.npz`` per seed and dataset; finished files are skipped. The raw MC samples are not
stored, only the criteria computed from them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from probly.selective_prediction import SelectivePredictor, ThresholdSelector
from sgr_experiment.data import load_cifar10, normalize
from sgr_experiment.loaders import load_base, load_dropout
from sgr_experiment.shift import (
    CORRUPTIONS,
    OOD_DATASETS,
    SEVERITIES,
    corrupt,
    corrupted_name,
    load_cifar100_test,
    load_svhn_test,
)
from sgr_experiment.uncertainty import decompose, predicted_class_variance
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, run_dir, seed_everything

ALL_DATASETS = ["cifar10", *OOD_DATASETS, *(corrupted_name(c, s) for c in CORRUPTIONS for s in SEVERITIES)]
CORRUPTION_SEED = 0
DOWNLOAD_HELP = """
Download of {name} failed ({err}).
The servers' certificates sometimes make torchvision fail on Windows. Download the file by hand (PowerShell) and
re-run the same command; torchvision then only verifies the MD5 and extracts it.

  CIFAR-100 (put it into {data_dir}, the extracted folder cifar-100-python/ is created on the next run):
    curl.exe -L -k --retry 10 --retry-all-errors -C - -o {data_dir}\\cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    expected MD5 of the archive: eb9058c3a382ffc7106e4002c42a8d85
    (the extracted test file has MD5 f0ef6b0ae62326f3e7ffdfab6717acfc)

  SVHN test split (put it into {data_dir}):
    curl.exe -L -k --retry 10 --retry-all-errors -C - -o {data_dir}\\test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
    expected MD5: eb5a983be6a315427106f1b164d9cef3
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--p", type=float, default=0.5, help="Dropout probability used in stage dropout.")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--subset", type=int, default=None, help="Use only the first N images of each dataset (smoke tests).")
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


def load_dataset(name: str, data_dir: Path, clean: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 images ``(n, 3, 32, 32)`` and labels (``-1`` for OOD) of a dataset; corruptions are built on the CPU."""
    x_clean, y_clean = clean
    if name == "cifar10":
        return x_clean, y_clean
    if name in OOD_DATASETS:
        try:
            x = load_svhn_test(data_dir) if name == "svhn" else load_cifar100_test(data_dir)
        except (OSError, RuntimeError) as err:
            raise SystemExit(DOWNLOAD_HELP.format(name=name, err=err, data_dir=data_dir)) from err
        return x, torch.full((len(x),), -1, dtype=torch.long)
    corruption, sev = name.removeprefix("c10_").rsplit("_s", 1)
    gen = torch.Generator().manual_seed(CORRUPTION_SEED)
    return corrupt(x_clean, corruption, int(sev), gen), y_clean


def predict_dataset(
    x_u8: torch.Tensor,
    base: torch.nn.Module,
    model: torch.nn.Module,
    predictor: SelectivePredictor,
    captured: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Softmax of both models and the five MC criteria for uint8 images."""
    n = len(x_u8)
    out = {k: np.zeros((n, 10), dtype=np.float32) for k in ("softmax_base", "softmax_dropout", "mean_probs")}
    out.update({k: np.zeros(n, dtype=np.float32) for k in ("mc_maxprob", "mc_total", "mc_aleatoric", "mc_epistemic", "mc_variance")})
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            x = normalize(x_u8[sl].to(device))
            out["softmax_base"][sl] = torch.softmax(base(x), dim=-1).float().cpu().numpy()
            out["softmax_dropout"][sl] = torch.softmax(model(x), dim=-1).float().cpu().numpy()
            result = predictor.predict(x)
            out["mc_maxprob"][sl] = result.uncertainty.float().cpu().numpy()
            rep = captured["rep"]
            probs = rep.samples.probabilities.float()  # (num_samples, batch, 10)
            out["mean_probs"][sl] = probs.mean(0).cpu().numpy()
            for k, v in decompose(rep).items():
                out[f"mc_{k}"][sl] = v
            out["mc_variance"][sl] = predicted_class_variance(probs).cpu().numpy()
    return out


def main() -> None:
    """Write ``runs/seed{S}/shift/{dataset}.npz`` for every missing seed x dataset pair."""
    args = parse_args()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    x_clean, y_clean = load_cifar10(args.data_dir, train=False)
    if args.subset:
        x_clean, y_clean = x_clean[: args.subset], y_clean[: args.subset]
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for seed in args.seeds:
        shift_dir = run_dir(args.runs, seed) / "shift"
        shift_dir.mkdir(parents=True, exist_ok=True)
        todo = [d for d in args.datasets if not (shift_dir / f"{d}.npz").exists()]
        if not todo:
            print(f"seed {seed}: all requested datasets exist, skipping.")
            continue
        seed_everything(seed + 54321)
        base = load_base(run_dir(args.runs, seed) / "base.pt").to(device)
        model = load_dropout(run_dir(args.runs, seed) / "dropout.pt", p=args.p).to(device)
        predictor = SelectivePredictor(
            model, ThresholdSelector(float("inf")), representer_kwargs={"num_samples": args.num_samples}
        )
        # Spy on the representer so that the very same MC samples feed the probly criterion and the decomposition.
        captured: dict[str, object] = {}
        original = predictor.representer.represent

        def spy(*a: object, _original: object = original, **k: object) -> object:
            rep = _original(*a, **k)
            captured["rep"] = rep
            return rep

        predictor.representer.represent = spy  # ty: ignore[invalid-assignment]
        for name in todo:
            if name not in cache:
                x_u8, labels = load_dataset(name, args.data_dir, (x_clean, y_clean))
                if args.subset:
                    x_u8, labels = x_u8[: args.subset], labels[: args.subset]
                cache[name] = (x_u8, labels)
            x_u8, labels = cache[name]
            t0 = time.perf_counter()
            res = predict_dataset(x_u8, base, model, predictor, captured, device, args.batch_size)
            np.savez(shift_dir / f"{name}.npz", labels=labels.numpy().astype(np.int64), **res)
            msg = f"seed {seed} {name}: {len(labels)} images in {time.perf_counter() - t0:.0f}s"
            if name == "cifar10":
                msg += f", MC-mean acc {(res['mean_probs'].argmax(1) == labels.numpy()).mean():.4f}"
                check_against_predictions(run_dir(args.runs, seed), res, labels.numpy(), args.subset)
            print(msg, flush=True)


def check_against_predictions(seed_dir: Path, res: dict[str, np.ndarray], labels: np.ndarray, subset: int | None) -> None:
    """Warn if the MC-mean accuracy differs from ``predictions_dropout.npz`` (MC is random, so only roughly)."""
    path = seed_dir / "predictions_dropout.npz"
    if not path.exists():
        return
    ref = np.load(path)
    ref_mean = ref["mean_probs"][: len(labels)]
    ref_acc = (ref_mean.argmax(1) == ref["labels"][: len(labels)]).mean()
    acc = (res["mean_probs"].argmax(1) == labels).mean()
    tol = 0.005 if subset is None else 0.1
    if abs(acc - ref_acc) > tol:
        print(f"WARNING: MC-mean accuracy {acc:.4f} differs from predictions_dropout.npz ({ref_acc:.4f}).")


if __name__ == "__main__":
    main()
