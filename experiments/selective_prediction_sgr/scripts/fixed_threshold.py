"""Fixed thresholds c for ``ThresholdSelector(c)``: risk and coverage on the test half, raw and temperature scaled.

With the default criterion (1 - max probability) ``ThresholdSelector(c)`` is Chow's rule: abstaining costs c, an error
costs 1. This script answers which risk and coverage a given c gives on CIFAR-10, and which c each of the paper's
desired risks r* needs. Temperature scaling is fitted on the 5k selection half (NLL), everything is reported on the
5k test half, with the same seed x split pairs as ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate import PAPER, fmt, load_seed
import numpy as np

from sgr_experiment.metrics import apply_threshold, threshold_for_risk
from sgr_experiment.utils import EXPERIMENT_DIR

C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
# Criteria of the form 1 - max probability, for which c has the Chow meaning (not the paper variance).
CRITERIA = ["sr_base", "sr_dropout", "mc_probly"]
N_BINS = 15


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    return p.parse_args()


def load_probs(seed_dir: Path) -> dict[str, np.ndarray]:
    """Probabilities behind each criterion: base softmax, dropout softmax (eval mode), mean MC probabilities."""
    base = np.load(seed_dir / "predictions_base.npz")
    d = np.load(seed_dir / "predictions_dropout.npz")
    return {
        "sr_base": base["softmax"].astype(np.float64),
        "sr_dropout": d["softmax"].astype(np.float64),
        "mc_probly": d["mean_probs"].astype(np.float64),
    }


def scale(log_probs: np.ndarray, t: float) -> np.ndarray:
    """Softmax of ``log_probs / t``; log probabilities are logits up to a per-row constant."""
    z = log_probs / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(log_probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    """Mean negative log likelihood after temperature ``t``."""
    z = log_probs / t
    m = z.max(1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(1))
    return float(np.mean(lse - z[np.arange(len(labels)), labels]))


def fit_temperature(log_probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimizing the NLL, by golden-section search over log t in [log 0.05, log 20]."""
    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(log_probs, labels, np.exp(a)), nll(log_probs, labels, np.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(log_probs, labels, np.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(log_probs, labels, np.exp(b))
    return float(np.exp((lo + hi) / 2))


def ece(probs: np.ndarray, labels: np.ndarray) -> float:
    """Expected calibration error of the top-class confidence with equal-width bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    total = 0.0
    for b in range(N_BINS):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main() -> None:
    """Evaluate the fixed-c grid and the c needed per r*, raw and temperature scaled."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(
        (f.parent for f in args.runs.glob("seed*/predictions_dropout.npz") if (f.parent / "predictions_base.npz").exists()),
        key=lambda p: int(p.name[4:]),
    )
    if not dirs:
        msg = f"No finished seeds under {args.runs}."
        raise SystemExit(msg)
    n = len(np.load(dirs[0] / "predictions_dropout.npz")["labels"])
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    half = n // 2
    risks = [r for r, _, _ in PAPER]
    variants = ["raw", "ts"]

    # res[variant][criterion][key] -> list over seed x split pairs
    res = {v: {c: {} for c in CRITERIA} for v in variants}

    def add(v: str, c: str, key: str, value: float) -> None:
        res[v][c].setdefault(key, []).append(value)

    for seed_dir in dirs:
        info = load_seed(seed_dir)
        labels = np.load(seed_dir / "predictions_dropout.npz")["labels"]
        probs = load_probs(seed_dir)
        for perm in splits:
            sel, test = perm[:half], perm[half:]
            for c in CRITERIA:
                loss = info["losses"][c]
                logp = np.log(np.clip(probs[c], 1e-30, None))
                t = fit_temperature(logp[sel], labels[sel])
                add("ts", c, "temperature", t)
                for v, p in (("raw", probs[c]), ("ts", scale(logp, t))):
                    crit = 1 - p.max(1)
                    add(v, c, "ece", ece(p[test], labels[test]))
                    for cc in C_GRID:
                        rk, cv = apply_threshold(crit[test], loss[test], cc)
                        add(v, c, f"risk@{cc}", rk)
                        add(v, c, f"cov@{cc}", cv)
                    for r in risks:
                        thr = threshold_for_risk(crit[sel], loss[sel], r)
                        rk, _ = apply_threshold(crit[test], loss[test], thr)
                        add(v, c, f"c@{r}", thr)
                        add(v, c, f"viol@{r}", float(rk > r) if np.isfinite(rk) else 0.0)

    names = {"raw": "raw probabilities", "ts": "temperature scaled (fitted on the selection half)"}
    md = [
        "# Fixed thresholds c (CIFAR-10)",
        "",
        f"{len(dirs)} seeds x {args.n_splits} random 5k/5k splits (same as `table.md`); mean +- std over all pairs.",
        "Accept if 1 - max prob <= c (`ThresholdSelector(c)`). Risk and coverage are on the 5k test half.",
        "",
        "## Calibration on the test half",
        "",
        "| criterion | ECE raw | ECE temperature scaled | temperature |",
        "|---|---|---|---|",
    ]
    for c in CRITERIA:
        md.append(
            f"| {c} | {fmt(res['raw'][c]['ece'])} | {fmt(res['ts'][c]['ece'])} | {fmt(res['ts'][c]['temperature'], 3)} |"
        )
    csv_rows = []
    for v in variants:
        md += ["", f"## Fixed c, {names[v]}", ""]
        cols = ["c"] + [f"{c} {k}" for c in CRITERIA for k in ("risk", "cov")]
        md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for cc in C_GRID:
            row = [f"{cc:g}"]
            for c in CRITERIA:
                row += [fmt(res[v][c][f"risk@{cc}"]), fmt(res[v][c][f"cov@{cc}"])]
                csv_rows.append([v, "fixed_c", c, f"{cc:g}", fmt(res[v][c][f"risk@{cc}"]), fmt(res[v][c][f"cov@{cc}"])])
            md.append("| " + " | ".join(row) + " |")
    for v in variants:
        md += [
            "",
            f"## c needed for r*, {names[v]}",
            "",
            "c is the threshold picked on the selection half for empirical risk <= r* (the held-out column of",
            "`table.md`); `violated` is the share of pairs whose test-half risk exceeds r*. SGR would bound this share",
            "by delta = 0.001 at the price of a smaller c.",
            "",
        ]
        cols = ["r*"] + [f"{c} {k}" for c in CRITERIA for k in ("c", "violated")]
        md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in risks:
            row = [f"{r:.2f}"]
            for c in CRITERIA:
                viol = np.mean(res[v][c][f"viol@{r}"])
                row += [fmt(res[v][c][f"c@{r}"]), f"{viol:.0%}"]
                csv_rows.append([v, "c_for_risk", c, f"{r:.2f}", fmt(res[v][c][f"c@{r}"]), f"{viol:.4f}"])
            md.append("| " + " | ".join(row) + " |")

    (args.out / "fixed_threshold.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (args.out / "fixed_threshold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["probabilities", "table", "criterion", "c_or_r_star", "risk_or_c", "coverage_or_violated"])
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"Wrote {args.out / 'fixed_threshold.md'} and {args.out / 'fixed_threshold.csv'}")


if __name__ == "__main__":
    main()
