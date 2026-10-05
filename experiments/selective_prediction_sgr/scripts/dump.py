"""Dump deterministic and MC-dropout predictions on the CIFAR-10 test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from probly.selective_prediction import SelectivePredictor, ThresholdSelector
from sgr_experiment.data import load_test_tensors
from sgr_experiment.loaders import load_base, load_dropout
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, run_dir, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--p", type=float, default=0.5, help="Dropout probability used in stage dropout.")
    p.add_argument("--batch-size", type=int, default=250)
    return p.parse_args()


def main() -> None:
    """Write ``predictions_base.npz`` and ``predictions_dropout.npz`` for one seed."""
    args = parse_args()
    device = get_device()
    out = run_dir(args.runs, args.seed)
    seed_everything(args.seed + 12345)
    x_all, y_all = load_test_tensors(args.data_dir)
    n = len(y_all)

    base = load_base(out / "base.pt").to(device)
    base_softmax = np.zeros((n, 10), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(start + args.batch_size, n))
            base_softmax[sl] = torch.softmax(base(x_all[sl].to(device)), dim=-1).float().cpu().numpy()
    np.savez(out / "predictions_base.npz", labels=y_all.numpy(), softmax=base_softmax)
    print(f"seed {args.seed}: base acc {(base_softmax.argmax(1) == y_all.numpy()).mean():.4f}")
    del base

    model = load_dropout(out / "dropout.pt", p=args.p).to(device)

    predictor = SelectivePredictor(
        model, ThresholdSelector(float("inf")), representer_kwargs={"num_samples": args.num_samples}
    )
    # Spy on the representer so that the very same MC samples feed the probly criterion and the raw dump.
    captured: dict[str, object] = {}
    original = predictor.representer.represent

    def spy(*a: object, **k: object) -> object:
        rep = original(*a, **k)
        captured["rep"] = rep
        return rep

    predictor.representer.represent = spy  # ty: ignore[invalid-assignment]

    softmax = np.zeros((n, 10), dtype=np.float32)
    mc_probs = np.zeros((args.num_samples, n, 10), dtype=np.float16)
    criterion = np.zeros(n, dtype=np.float32)
    mean_probs = np.zeros((n, 10), dtype=np.float32)
    var_pred = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(start + args.batch_size, n))
            x = x_all[sl].to(device)
            softmax[sl] = torch.softmax(model(x), dim=-1).float().cpu().numpy()
            result = predictor.predict(x)
            criterion[sl] = result.uncertainty.float().cpu().numpy()
            probs = captured["rep"].samples.probabilities.float()  # (num_samples, batch, 10)
            mc_probs[:, sl] = probs.cpu().numpy().astype(np.float16)
            mean = probs.mean(0)
            mean_probs[sl] = mean.cpu().numpy()
            pred = mean.argmax(-1)
            pred_probs = probs.gather(-1, pred.expand(probs.shape[0], -1).unsqueeze(-1)).squeeze(-1)
            var_pred[sl] = pred_probs.var(0, unbiased=False).cpu().numpy()
            print(f"dumped {sl.stop}/{n}", flush=True)

    labels = y_all.numpy()
    np.savez(
        out / "predictions_dropout.npz",
        labels=labels,
        softmax=softmax,
        mc_probs=mc_probs,
        criterion_probly=criterion,
        mean_probs=mean_probs,
        criterion_variance=var_pred,
    )
    print(
        f"seed {args.seed}: dropout deterministic acc {(softmax.argmax(1) == labels).mean():.4f}, "
        f"MC-mean acc {(mean_probs.argmax(1) == labels).mean():.4f}"
    )


if __name__ == "__main__":
    main()
