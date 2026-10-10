"""Compare selective prediction criteria with Table 1 of Geifman and El-Yaniv (2017)."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import warnings

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.metrics.selective_prediction import coverage_at_risk, risk_coverage_curve  # noqa: E402
from probly.selective_prediction import CoverageSelector, SGRSelector  # noqa: E402
from sgr_experiment.metrics import apply_threshold, threshold_for_risk  # noqa: E402
from sgr_experiment.uncertainty import one_minus_max  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

# Paper Table 1: desired risk r*, test risk, test coverage.
PAPER = [
    (0.01, 0.0092, 0.7856),
    (0.02, 0.0149, 0.8466),
    (0.03, 0.0261, 0.8966),
    (0.04, 0.0380, 0.9318),
    (0.05, 0.0486, 0.9596),
    (0.06, 0.0572, 0.9784),
]
PAPER_ACC = 0.9354
# Explicit family: weight 100 in "Fira Sans" resolves to the hairline FiraSans-Two, not to Thin.
THIN_FONT = ["Fira Sans Thin", "Fira Sans"]
CRITERIA = ["sr_base", "sr_dropout", "mc_probly", "mc_paper_variance"]
LABELS = {
    "sr_base": "SR, base model",
    "sr_dropout": "SR, dropout model",
    "mc_probly": "MC dropout, probly (1 - max mean prob)",
    "mc_paper_variance": "MC dropout, paper variance",
}
COLORS = {"sr_base": "#16a085", "sr_dropout": "#9b59b6", "mc_probly": "#1e88e5", "mc_paper_variance": "#ff0d57"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of probly's SGRSelector.")
    return p.parse_args()


def setup_fonts() -> bool:
    """Use Fira Sans if installed, else the default font, without warnings."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    try:
        font_manager.findfont("Fira Sans", fallback_to_default=False)
    except (ValueError, KeyError):
        return False
    plt.rcParams["font.family"] = ["Fira Sans", "DejaVu Sans"]
    return True


def load_seed(seed_dir: Path) -> dict:
    """Load both prediction files of one seed and derive criteria and losses."""
    base = np.load(seed_dir / "predictions_base.npz")
    d = np.load(seed_dir / "predictions_dropout.npz")
    labels = d["labels"]
    np.testing.assert_array_equal(labels, base["labels"])
    mc = d["mc_probs"].astype(np.float32)
    mean = d["mean_probs"].astype(np.float64)
    probly = d["criterion_probly"].astype(np.float64)
    np.testing.assert_allclose(probly, 1 - mean.max(1), atol=1e-4)
    np.testing.assert_allclose(probly, 1 - mc.mean(0).max(1), atol=5e-3)
    pred = mean.argmax(1)
    losses = {
        "sr_base": (base["softmax"].argmax(1) != labels).astype(np.float64),
        "sr_dropout": (d["softmax"].argmax(1) != labels).astype(np.float64),
        "mc_probly": (pred != labels).astype(np.float64),
    }
    losses["mc_paper_variance"] = losses["mc_probly"]
    criteria = {
        "sr_base": one_minus_max(base["softmax"]),
        "sr_dropout": one_minus_max(d["softmax"]),
        "mc_probly": probly,
        "mc_paper_variance": d["criterion_variance"].astype(np.float64),
    }
    return {
        "criteria": criteria,
        "losses": losses,
        "acc_base": 1 - losses["sr_base"].mean(),
        "acc_dropout_det": 1 - losses["sr_dropout"].mean(),
        "acc_dropout_mc": 1 - losses["mc_probly"].mean(),
    }


def fmt(values: np.ndarray, digits: int = 4) -> str:
    """Mean +- std over finite entries."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return f"{v.mean():.{digits}f} +- {v.std():.{digits}f}"


def main() -> None:
    """Build the table and the risk-coverage plot."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(
        (f.parent for f in args.runs.glob("seed*/predictions_dropout.npz") if (f.parent / "predictions_base.npz").exists()),
        key=lambda p: int(p.name[4:]),
    )
    if not dirs:
        msg = f"No finished seeds (predictions_base.npz and predictions_dropout.npz) under {args.runs}."
        raise SystemExit(msg)
    seeds = [int(d.name[4:]) for d in dirs]
    data = [load_seed(d) for d in dirs]
    n = len(data[0]["losses"]["sr_base"])
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    half = n // 2

    risks = [r for r, _, _ in PAPER]
    matched = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    held_risk = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    held_cov = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    sgr_risk = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    sgr_cov = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    sgr_viol = {c: np.zeros((len(risks), 0)) for c in CRITERIA}
    cs_cov = {c: np.zeros((len(PAPER), 0)) for c in CRITERIA}
    cs_risk = {c: np.zeros((len(PAPER), 0)) for c in CRITERIA}
    for d in data:
        for perm in splits:
            sel, test = perm[:half], perm[half:]
            for c in CRITERIA:
                crit, loss = d["criteria"][c], d["losses"][c]
                m = np.array([coverage_at_risk(crit[test], loss[test], pr) for _, pr, _ in PAPER])
                hr, hc = [], []
                for r in risks:
                    thr = threshold_for_risk(crit[sel], loss[sel], r)
                    rk, cv = apply_threshold(crit[test], loss[test], thr)
                    hr.append(rk)
                    hc.append(cv)
                sr, sc, sv = [], [], []
                for r in risks:
                    with warnings.catch_warnings():  # uncertified: threshold -inf, accepts nothing
                        warnings.simplefilter("ignore", UserWarning)
                        thr = SGRSelector(r, args.delta).calibrate(crit[sel], loss[sel]).threshold
                    rk, cv = apply_threshold(crit[test], loss[test], thr)
                    sr.append(rk)
                    sc.append(cv)
                    sv.append(float(rk > r + 1e-12) if np.isfinite(rk) else 0.0)
                cr, cc = [], []
                for _, _, paper_cov in PAPER:
                    thr = CoverageSelector(paper_cov).calibrate(crit[sel]).threshold
                    rk, cv = apply_threshold(crit[test], loss[test], thr)
                    cr.append(rk)
                    cc.append(cv)
                for store, vals in ((sgr_risk, sr), (sgr_cov, sc), (sgr_viol, sv), (cs_risk, cr), (cs_cov, cc)):
                    store[c] = np.column_stack([store[c], vals])
                matched[c] = np.column_stack([matched[c], m])
                held_risk[c] = np.column_stack([held_risk[c], hr])
                held_cov[c] = np.column_stack([held_cov[c], hc])

    header = ["r_star", "paper_test_risk", "paper_test_coverage"]
    for c in CRITERIA:
        header += [f"{c}_cov_at_paper_risk", f"{c}_heldout_risk", f"{c}_heldout_coverage"]
    rows = []
    for i, (r, pr, pc) in enumerate(PAPER):
        row = [f"{r:.2f}", f"{pr:.4f}", f"{pc:.4f}"]
        for c in CRITERIA:
            row += [fmt(matched[c][i]), fmt(held_risk[c][i]), fmt(held_cov[c][i])]
        rows.append(row)
    with (args.out / "table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    sgr_header = ["r_star"] + [f"{c}_{k}" for c in CRITERIA for k in ("sgr_risk", "sgr_coverage", "sgr_violation_share")]
    sgr_rows = []
    cs_header = ["paper_test_coverage", "paper_test_risk"] + [f"{c}_{k}" for c in CRITERIA for k in ("cs_coverage", "cs_risk")]
    cs_rows = []
    for i, (r, pr, pc) in enumerate(PAPER):
        sgr_rows.append([f"{r:.2f}"] + [x for c in CRITERIA for x in (fmt(sgr_risk[c][i]), fmt(sgr_cov[c][i]), f"{np.mean(sgr_viol[c][i]) * 100:.0f}%")])
        cs_rows.append([f"{pc:.4f}", f"{pr:.4f}"] + [x for c in CRITERIA for x in (fmt(cs_cov[c][i]), fmt(cs_risk[c][i]))])
    for name, head, body in (("table_sgr.csv", sgr_header, sgr_rows), ("table_coverage.csv", cs_header, cs_rows)):
        with (args.out / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(body)

    md = [
        "# Table 1 reproduction (CIFAR-10)",
        "",
        f"{len(seeds)} seeds x {args.n_splits} random 5k/5k splits; entries are mean +- std over all seed x split pairs.",
        "`cov@risk` is the coverage on the 5k test half at the paper's test risk. `held-out` picks the threshold on",
        "the 5k selection half for r* (empirical risk <= r*) and reports the resulting risk and coverage on the test half.",
        "",
    ]
    cols = ["r*", "paper risk", "paper cov"]
    for c in CRITERIA:
        cols += [f"{c} cov@risk", f"{c} held-out risk", f"{c} held-out cov"]
    md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += [
        "",
        "## SGR guarantee (probly SGRSelector)",
        "",
        f"Threshold from the selection half with SGR (delta {args.delta}) for r*; risk, coverage and the share of splits",
        "with test risk > r* on the test half. An uncertified selector accepts nothing (risk n/a, coverage 0).",
        "",
    ]
    cols = ["r*"] + [f"{c} {k}" for c in CRITERIA for k in ("risk", "cov", "viol")]
    md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(r) + " |" for r in sgr_rows]
    md += [
        "",
        "## Coverage target (probly CoverageSelector)",
        "",
        "Label-free: the threshold is the split-conformal quantile of the criterion on the selection half for the paper's",
        "test coverage; realized coverage and risk on the test half, next to the paper's risk.",
        "",
    ]
    cols = ["paper cov", "paper risk"] + [f"{c} {k}" for c in CRITERIA for k in ("cov", "risk")]
    md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(r) + " |" for r in cs_rows]
    md += ["", "## Base accuracy", "", f"Paper: {PAPER_ACC * 100:.2f}%", "", "| seed | base | dropout model, deterministic | dropout model, MC mean |", "|---|---|---|---|"]
    for s, d in zip(seeds, data, strict=True):
        md.append(
            f"| {s} | {d['acc_base'] * 100:.2f}% | {d['acc_dropout_det'] * 100:.2f}% | {d['acc_dropout_mc'] * 100:.2f}% |"
        )
    mean_acc = [np.mean([d[k] for d in data]) * 100 for k in ("acc_base", "acc_dropout_det", "acc_dropout_mc")]
    md.append("| mean | " + " | ".join(f"{a:.2f}%" for a in mean_acc) + " |")
    (args.out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plot(args.out / "risk_coverage.png", data)
    print("\n".join(md))
    print(f"Wrote {args.out / 'table.md'}, {args.out / 'table.csv'}, {args.out / 'risk_coverage.png'}")


def plot(path: Path, data: list[dict]) -> None:
    """Risk-coverage curves on the full test set with the paper's points overlaid."""
    setup_fonts()
    grid = np.linspace(0.7, 1.0, 601)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in CRITERIA:
        ys = []
        for i, d in enumerate(data):
            cov, rsk, _ = risk_coverage_curve(d["criteria"][c], d["losses"][c])
            y = np.interp(grid, cov, rsk)
            ys.append(y)
            ax.plot(grid, y, color=COLORS[c], lw=0.6, alpha=0.3)
        ax.plot(grid, np.mean(ys, axis=0), color=COLORS[c], lw=2, label=LABELS[c])
    ax.scatter(
        [pc for _, _, pc in PAPER], [pr for _, pr, _ in PAPER], marker="D", color="black", zorder=5, label="paper (SGR)"
    )
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("coverage", fontweight="semibold")
    ax.set_ylabel("selective risk", fontweight="semibold")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(THIN_FONT)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
