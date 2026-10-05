"""Train, dump and evaluate all seeds; finished steps are skipped, so it can be re-run after an interruption."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from sgr_experiment.utils import EXPERIMENT_DIR, default_workers, run_dir

SCRIPTS = Path(__file__).resolve().parent


def run(script: str, *args: object) -> None:
    """Run a sibling script in a subprocess with the current interpreter."""
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    """Loop over seeds, then optionally evaluate."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--finetune-lr", type=float, default=0.01)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    for seed in a.seeds:
        rd = run_dir(a.runs, seed)
        extra = ["--subset", a.subset] if a.subset else []
        common = ["--seed", seed, "--out", a.runs, "--data-dir", a.data_dir, "--workers", a.workers, *extra]
        # train.py skips a stage whose final weights exist, and resumes an interrupted one from its checkpoint.
        run("train.py", "--stage", "base", "--epochs", a.epochs, *common)
        run("train.py", "--stage", "dropout", "--epochs", a.finetune_epochs, "--lr", a.finetune_lr, *common)
        if (rd / "predictions_base.npz").exists() and (rd / "predictions_dropout.npz").exists():
            print(f"seed {seed}: predictions exist, skipping dump.")
        else:
            run("dump.py", "--seed", seed, "--runs", a.runs, "--data-dir", a.data_dir, "--num-samples", a.num_samples)
    if a.evaluate:
        run("evaluate.py", "--runs", a.runs, "--out", a.out, "--n-splits", a.n_splits)


if __name__ == "__main__":
    main()
