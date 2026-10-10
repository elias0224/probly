"""Reproduction of Tables 3-6 (Sec. 5.3, ImageNet) of Geifman and El-Yaniv (2017), arXiv 1705.08500.

Reads the dumps of ``dump_imagenet.py``. Per model, task (top-1, top-5 loss) and criterion, ``--n-splits`` random
halvings of the validation set; the first half ("train" in the paper) picks the threshold for each r* of the paper's
table, the second half gives test risk and coverage. Rules: ``sgr`` (probly's ``SGRSelector``, delta 0.001 as in the
paper), ``emp`` (largest coverage with selection-half risk <= r*, no guarantee), and the sweep ``sgr_delta_{d}``
(``--deltas``, SGR at a larger delta). The paper's numbers should lie between ``sgr`` and ``emp`` (see the bound
check), and the sweep shows which delta reproduces them. ``cov @ paper risk`` is the largest test-half coverage whose risk is at most
the paper's test risk, ``risk @ paper cov`` the test-half risk at the paper's test coverage; both compare the
risk-coverage curves and do not depend on SGR.

The first section recomputes the bound B* (Lemma 3.1) from the paper's own train risk and train coverage. The second
lines our risk-coverage curves up with the paper's test points for the paper's criterion (SR), VGG-16 first: VGG-16 is
the comparison that counts, because torchvision's ResNet-50 is likely stronger than the paper's model.

Outputs in ``--out``: ``table.md``, ``table.csv``, ``risk_coverage.png`` (Fig. 2c style, with the paper's test
points) and ``coverage.png`` (test coverage against r*).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate import THIN_FONT, setup_fonts
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.metrics.selective_prediction import aurc, risk_coverage_curve  # noqa: E402

from sgr_experiment.imagenet import (  # noqa: E402
    CRITERIA,
    PAPER_DELTA,
    PAPER_M,
    PAPER_TABLE_NUMBER,
    PAPER_TABLES,
    TASKS,
    evaluate_unit,
    paper_bound_check,
    rules_with_sweep,
    splits,
    topk_losses,
)
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

BLUE, RED = "#1e88e5", "#ff0d57"
CRITERIA_PER_TASK = {"top1": ["sr", "mc_variance", "mc_maxprob"], "top5": ["sr", "top5_mass", "mc_variance", "mc_maxprob"]}
COLUMNS = [
    ("certified", "certified", "pct"),
    ("sel_risk", "train risk", "f"),
    ("sel_cov", "train cov", "f"),
    ("test_risk", "test risk", "f"),
    ("test_cov", "test cov", "f"),
    ("bound", "bound", "f"),
    ("viol", "test risk > r*", "pct"),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs" / "imagenet")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "imagenet")
    p.add_argument("--models", nargs="+", default=["vgg16", "resnet50"])
    p.add_argument("--n-splits", type=int, default=100)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=PAPER_DELTA)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.1], help="Larger deltas of the SGR sweep.")
    return p.parse_args()


def load(runs: Path, model: str) -> dict[str, np.ndarray] | None:
    """Arrays of one dump, or None if it does not exist."""
    path = runs / f"{model}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files if k != "probs"}


def mean(values: list[float]) -> float:
    """Mean over the finite values (NaN if there are none)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def cell(values: list[float], kind: str) -> str:
    """Markdown cell: mean (+- std for numbers), or a percentage."""
    m = mean(values)
    if not np.isfinite(m):
        return "n/a"
    if kind == "pct":
        return f"{m * 100:.0f}%"
    v = np.asarray(values, dtype=float)
    return f"{m:.4f} +- {v[np.isfinite(v)].std():.4f}"


def units(data: dict[str, np.ndarray], task: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Criterion name -> (criterion, top-k loss of the matching predictions) for the criteria in the dump."""
    out = {}
    for c in CRITERIA_PER_TASK[task]:
        key, pred_key, _ = CRITERIA[c]
        if key in data:
            out[c] = (data[key].astype(np.float64), topk_losses(data[pred_key], data["labels"], TASKS[task]))
    return out


def check_section(models: list[str]) -> list[str]:
    """Markdown: B* of the paper's train numbers at m = 25000, with delta / ceil(log2 m) and with the full delta."""
    steps = math.ceil(math.log2(PAPER_M))
    lines = [
        "## Check of the paper's Tables 3-6",
        "",
        f"B* (Lemma 3.1, Clopper-Pearson) of the paper's train risk and train coverage on m = {PAPER_M} selection "
        f"points, at delta {PAPER_DELTA} / {steps} as in Algorithm 1 and at the full delta {PAPER_DELTA}. A row is "
        "consistent with the paper's bound only if B* <= the paper's bound.",
        "",
        "| table | r* | accepted | errors | paper bound | B*, delta/15 | B*, delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for task in TASKS:
            for row in paper_bound_check(model, task):
                lines.append(
                    f"| {PAPER_TABLE_NUMBER[model, task]} ({model} {task}) | {row['r']:.2f} | {row['accepted']} | "
                    f"{row['errors']} | {row['paper_bound']:.4f} | {row['bound_split']:.4f} | {row['bound_full']:.4f} |"
                )
    return [*lines, ""]


def lineup_section(results: dict, full: dict, models: list[str]) -> list[str]:
    """Markdown: our test-half risk at the paper's test coverage (and coverage at its risk) for SR, per table row.

    This compares the risk-coverage curves only, independent of SGR and delta. VGG-16 comes first because it is the
    comparison that counts; torchvision's ResNet-50 is likely stronger than the paper's model.
    """
    lines = [
        "## Line-up with the paper's test points (SR, the paper's criterion)",
        "",
        "Mean over the test halves. Delta columns are ours minus the paper's. Rows where both deltas are small mean our "
        "model has the paper's risk-coverage curve at that point.",
        "",
    ]
    for model in sorted(models, key=lambda m: m != "vgg16"):
        for task in TASKS:
            if (model, task, "sr") not in results:
                continue
            res = results[model, task, "sr"]
            diffs = []
            lines += [
                f"### Table {PAPER_TABLE_NUMBER[model, task]}: {model} {task}, full-coverage risk {full[model, task, 'sr']['risk']:.4f}",
                "",
                "| r* | paper test cov | paper test risk | risk @ paper cov | delta risk | cov @ paper risk | delta cov |",
                "|---|---|---|---|---|---|---|",
            ]
            for r, _, _, p_risk, p_cov, _ in PAPER_TABLES[model, task]:
                risk, cov = mean(res[r, "risk_at_paper"]), mean(res[r, "cov_at_paper"])
                diffs.append(risk - p_risk)
                lines.append(
                    f"| {r:.2f} | {p_cov:.4f} | {p_risk:.4f} | {risk:.4f} | {risk - p_risk:+.4f} | {cov:.4f} | {cov - p_cov:+.4f} |"
                )
            lines += ["", f"Mean absolute delta risk {np.mean(np.abs(diffs)):.4f}.", ""]
    return lines


def evaluate(datas: dict[str, dict], args: argparse.Namespace) -> tuple[dict, dict]:
    """Results per ``(model, task, criterion)`` and the full-coverage numbers per ``(model, task, criterion)``."""
    results, full = {}, {}
    for model, data in datas.items():
        halves = splits(len(data["labels"]), args.n_splits, args.split_seed)
        for task in TASKS:
            for c, (crit, loss) in units(data, task).items():
                results[model, task, c] = evaluate_unit(crit, loss, PAPER_TABLES[model, task], halves, args.delta, args.rules)
                full[model, task, c] = {"risk": float(loss.mean()), "aurc": float(aurc(crit, loss))}
    return results, full


def write(out: Path, results: dict, full: dict, datas: dict, args: argparse.Namespace) -> None:
    """``table.md`` and ``table.csv``."""
    n = {m: len(d["labels"]) for m, d in datas.items()}
    lines = [
        "# Tables 3-6 reproduction (Geifman and El-Yaniv 2017, ImageNet)",
        "",
        f"{args.n_splits} random selection/test halvings of the validation set, delta {args.delta}. train = selection "
        "half. Risk, bound and violation columns over the splits where the rule certified a threshold, coverage "
        "columns over all splits (0 when nothing is certified). cov @ paper risk: largest test-half coverage at risk "
        "<= the paper's test risk; risk @ paper cov: test-half risk at the paper's test coverage.",
        "",
        *check_section(["vgg16", "resnet50"]),
        *lineup_section(results, full, list(datas)),
    ]
    for model, data in datas.items():
        meta_path = args.runs / f"{model}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        acc = meta.get("accuracy", {})
        lines += [
            f"## {model}",
            "",
            f"{n[model]} validation images, weights `{meta.get('weights', '?')}`, top-1 acc {acc.get('top1', float('nan')):.4f}, "
            f"top-5 acc {acc.get('top5', float('nan')):.4f}"
            + (f", MC mean top-1 {acc['mc_top1']:.4f} / top-5 {acc['mc_top5']:.4f} ({meta['mc_samples']} samples)" if "mc_top1" in acc else "")
            + ".",
            "",
        ]
        for task in TASKS:
            for c in CRITERIA_PER_TASK[task]:
                if (model, task, c) not in results:
                    continue
                res = results[model, task, c]
                f = full[model, task, c]
                lines += [
                    f"### Table {PAPER_TABLE_NUMBER[model, task]}: {model} {task}, {CRITERIA[c][2]} (`{c}`)",
                    "",
                    f"Full-coverage risk {f['risk']:.4f}, AURC {f['aurc']:.4f} (all {n[model]} images).",
                    "",
                    "| r* | rule | " + " | ".join(h for _, h, _ in COLUMNS) + " | paper test risk | paper test cov | cov @ paper risk | risk @ paper cov |",
                    "|" + "---|" * (len(COLUMNS) + 6),
                ]
                for r, _, _, p_risk, p_cov, _ in PAPER_TABLES[model, task]:
                    for i, rule in enumerate(args.rules):
                        cells = [cell(res[rule, r, k], kind) for k, _, kind in COLUMNS]
                        extra = (
                            [f"{p_risk:.4f}", f"{p_cov:.4f}", cell(res[r, "cov_at_paper"], "f"), cell(res[r, "risk_at_paper"], "f")]
                            if i == 0
                            else ["", "", "", ""]
                        )
                        lines.append(f"| {r:.2f} | {rule} | " + " | ".join(cells + extra) + " |")
                lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    with (out / "table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "criterion", "rule", "r_star", *(f"{k}_{s}" for k, _, _ in COLUMNS for s in ("mean", "std")), "paper_test_risk", "paper_test_cov", "cov_at_paper_mean", "risk_at_paper_mean"])
        for (model, task, c), res in results.items():
            for r, _, _, p_risk, p_cov, _ in PAPER_TABLES[model, task]:
                for rule in args.rules:
                    vals = []
                    for k, _, _ in COLUMNS:
                        v = np.asarray(res[rule, r, k], dtype=float)
                        v = v[np.isfinite(v)]
                        vals += [f"{v.mean():.6g}", f"{v.std():.6g}"] if v.size else ["", ""]
                    w.writerow([model, task, c, rule, r, *vals, p_risk, p_cov, f"{mean(res[r, 'cov_at_paper']):.6g}", f"{mean(res[r, 'risk_at_paper']):.6g}"])


def style_axes(ax: plt.Axes, has_fira: bool) -> None:
    """Grid and thin tick labels."""
    ax.grid(alpha=0.3)
    if has_fira:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily(THIN_FONT)


def plot_curves(path: Path, datas: dict) -> None:
    """Risk-coverage curves on the whole validation set, Fig. 2c style: SR blue, MC dropout red, top-1 dashed, top-5 solid.

    Black markers are the paper's (test coverage, test risk) points from Tables 3-6.
    """
    has_fira = setup_fonts()
    fig, axes = plt.subplots(1, len(datas), figsize=(5.2 * len(datas), 4.2), squeeze=False, sharey=True)
    for ax, (model, data) in zip(axes[0], datas.items(), strict=True):
        for task, ls in (("top1", "--"), ("top5", "-")):
            u = units(data, task)
            for c, color in (("sr", BLUE), ("mc_variance", RED)):
                if c in u:
                    cov, risk, _ = risk_coverage_curve(*u[c])
                    ax.plot(cov, risk, color=color, ls=ls, lw=1.4, label=f"{'SR' if c == 'sr' else 'MC dropout'}, {task}")
            pts = np.array([(row[4], row[3]) for row in PAPER_TABLES[model, task]])
            ax.plot(pts[:, 0], pts[:, 1], "o" if task == "top5" else "s", color="black", ms=4, mfc="none", label=f"paper, {task}")
        ax.set_title(model)
        ax.set_xlabel("Coverage", fontweight="semibold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.35)
        style_axes(ax, has_fira)
    axes[0][0].set_ylabel("Risk", fontweight="semibold")
    axes[0][0].legend(loc="upper left", fontsize=8)  # the VGG-16 panel is the one with the MC-dropout curves
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_coverage(path: Path, results: dict, models: list[str]) -> None:
    """SGR test coverage against r* (SR, and top-5 mass for top-5) next to the paper's test coverage."""
    has_fira = setup_fonts()
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.8 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, task in enumerate(TASKS):
            ax = axes[i][j]
            rows = PAPER_TABLES[model, task]
            rs = [row[0] for row in rows]
            ax.plot(rs, [row[4] for row in rows], color="black", marker="o", lw=1.5, label="paper")
            for c, color in (("sr", BLUE), ("top5_mass", "#16a085"), ("mc_variance", RED)):
                if (model, task, c) not in results:
                    continue
                res = results[model, task, c]
                ax.plot(rs, [mean(res["sgr", r, "test_cov"]) for r in rs], color=color, marker=".", lw=1.4, label=f"{c}, sgr")
                ax.plot(rs, [mean(res["emp", r, "test_cov"]) for r in rs], color=color, ls=":", lw=1.2, label=f"{c}, emp")
                if c == "sr":  # the delta sweep for the paper's criterion only, to keep the panels readable
                    sweep = sorted({k[0] for k in res if isinstance(k[0], str) and k[0].startswith("sgr_delta_")})
                    for name, ls in zip(sweep, ("--", "-.", (0, (5, 1)), (0, (1, 1))), strict=False):
                        ax.plot(rs, [mean(res[name, r, "test_cov"]) for r in rs], color=color, ls=ls, lw=1.0, label=f"sr, {name}")
            ax.set_title(f"Table {PAPER_TABLE_NUMBER[model, task]}: {model} {task}", fontsize=10)
            ax.set_xlabel("Desired risk r*", fontweight="semibold")
            ax.set_ylabel("Test coverage", fontweight="semibold")
            style_axes(ax, has_fira)
            ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Evaluate the dumps and write the tables and figures."""
    args = parse_args()
    args.rules = rules_with_sweep(args.deltas)
    datas = {m: d for m in args.models if (d := load(args.runs, m)) is not None}
    if not datas:
        msg = f"no dumps for {args.models} in {args.runs}; run dump_imagenet.py first"
        raise SystemExit(msg)
    args.out.mkdir(parents=True, exist_ok=True)
    results, full = evaluate(datas, args)
    write(args.out, results, full, datas, args)
    plot_curves(args.out / "risk_coverage.png", datas)
    plot_coverage(args.out / "coverage.png", results, list(datas))
    print((args.out / "table.md").read_text())


if __name__ == "__main__":
    main()
