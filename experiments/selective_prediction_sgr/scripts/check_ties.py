"""Count ties of the softmax response criterion in the stored predictions (float32 saturation check).

Prints per seed the share of test points whose stored float32 max softmax is exactly 1.0, the error count inside that
tie block, and the share of tied criterion values for ``1 - max`` versus the tie-free :func:`one_minus_max`.
"""

from __future__ import annotations

import numpy as np

from sgr_experiment.uncertainty import one_minus_max
from sgr_experiment.utils import EXPERIMENT_DIR


def tie_share(crit: np.ndarray) -> float:
    """Share of values that are not unique."""
    _, counts = np.unique(crit, return_counts=True)
    return float(counts[counts > 1].sum() / crit.size)


def main() -> None:
    """Print the tie statistics of every seed's base and dropout softmax."""
    for seed_dir in sorted((EXPERIMENT_DIR / "runs").glob("seed*")):
        for name in ("predictions_base.npz", "predictions_dropout.npz"):
            f = seed_dir / name
            if not f.exists():
                continue
            d = np.load(f)
            p, labels = d["softmax"], d["labels"]
            sat = p.max(1) == 1.0
            errors = int((p.argmax(1)[sat] != labels[sat]).sum())
            print(
                f"{seed_dir.name} {name}: max==1.0 {sat.mean():.3f} ({errors} errors), "
                f"ties 1-max {tie_share(1 - p.astype(np.float64).max(1)):.3f}, tie-free {tie_share(one_minus_max(p)):.3f}"
            )


if __name__ == "__main__":
    main()
