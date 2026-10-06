"""Count ties of the softmax response criterion in the stored predictions (float32 resolution check).

Float32 values just below 1.0 are spaced 2**-24 apart, so ``1 - max`` only takes a few distinct values for confident
predictions. Prints per seed the share of tied criterion values for ``1 - max`` versus the tie-free
:func:`one_minus_max`, plus the size of the lowest-uncertainty tie block of ``1 - max`` and the errors inside it.
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
            naive = 1 - p.astype(np.float64).max(1)
            block = naive == naive.min()
            errors = int((p.argmax(1)[block] != labels[block]).sum())
            print(
                f"{seed_dir.name} {name}: ties 1-max {tie_share(naive):.3f}, tie-free {tie_share(one_minus_max(p)):.3f}, "
                f"lowest block {block.mean():.3f} ({errors} errors)"
            )


if __name__ == "__main__":
    main()
