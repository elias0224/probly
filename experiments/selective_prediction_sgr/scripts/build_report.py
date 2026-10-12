"""Build the readable HTML (and PDF) report of the selective-prediction experiment.

Order: summary, setup, implementation differences, replication of Geifman and El-Yaniv 2017 (CIFAR-10 Table 1, ImageNet
Tables 3-6 and Fig. 2c), the paper's printed bounds, method comparison, threshold rules, deep ensemble + SGR, other
experiments (calibration, OOD, shift), appendix with raw ids. The ImageNet sections need
``imagenet_table.py`` to have run first (they are skipped with a note otherwise). All numbers are read from the result csv files; the prose lives in ``report_text.py``. Tables in the main sections
are short on purpose, long ones are folded or in the appendix.

Usage:
    uv run python scripts/build_report.py --results results --pdf
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import build_summary as bs
from build_summary import img_tag, num, read_csv
from evaluate import setup_fonts
from evaluate_shift import CRITERIA, RULE_LABELS
import matplotlib.pyplot as plt
import numpy as np
import report_text as rt
from sgr_experiment.imagenet import PAPER_TABLE_NUMBER, PAPER_TABLES, paper_bound_check
from sgr_variants import paper_check_rows

KEY_R = [0.01, 0.02, 0.05]
KEY_METHODS = ["sr_base", "mc_maxprob", "ens_maxprob", "ens_epistemic", "dropout_scratch_sr"]
BLUE, RED = "#1e88e5", "#ff0d57"  # shapiq colors
PAPER_COLOR = "black"  # the paper in every figure
ENS_COLOR = BLUE
ORANGE, PURPLE = "#e65100", "#7e57c2"
SOFT_COLOR = "#16a085"
MC_PASSES = 100

REPORT_CSS = (
    bs.CSS
    + """
h3 { font-size:15px; font-weight:600; margin:18px 0 2px; }
section { margin-bottom:8px; }
dl.gloss { display:grid; grid-template-columns:150px 1fr; gap:2px 12px; font-size:14px; margin:8px 0; }
dl.gloss dt { font-weight:600; } dl.gloss dd { margin:0; color:var(--mut); }
td small, th small { color:var(--mut); font-size:11px; }
.d { color:var(--mut); font-size:13px; }
.g { color:#1b7f5c; font-weight:600; } .n { color:var(--mut); }
details { margin:10px 0; } summary { cursor:pointer; color:var(--mut); }
details table { font-size:12px; }
table.txt th, table.txt td { text-align:left; }
@media print { section { break-inside:avoid-page; } img, table { break-inside:avoid; } h2, h3 { break-after:avoid; } }
"""
)


def pct(x: float, d: int = 1) -> str:
    """Format a share as percent text."""
    return "n/a" if math.isnan(x) else f"{100 * x:.{d}f}%"


def pts(x: float) -> str:
    """Format a coverage difference in percentage points."""
    return "n/a" if math.isnan(x) else f"{100 * x:.1f}"


def name(crit: str) -> str:
    """Display name of a criterion."""
    return bs.crit_style(crit)["label"]


def table(header: list[str], rows: list[list[str]]) -> str:
    """Small HTML table."""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def raw_table(rows: list[dict[str, str]], cols: list[str] | None = None) -> str:
    """Folded-appendix table of raw csv rows (4 significant digits)."""
    if not rows:
        return "<p>missing</p>"
    cols = cols or list(rows[0])

    def cell(v: str) -> str:
        x = num(v)
        return html.escape(v) if math.isnan(x) else f"{x:.4g}"

    return table(
        [html.escape(c) for c in cols], [[cell(r.get(c, "")) if c != "criterion" else r.get(c, "") for c in cols] for r in rows]
    )


def row_at(rows: list[dict[str, str]], r: float, **kw: str) -> dict[str, str]:
    """First row with r_star == r and the given column values."""
    for x in rows:
        if abs(num(x.get("r_star")) - r) < 1e-9 and all(x.get(k) == v for k, v in kw.items()):
            return x
    return {}


class Data:
    """All csv inputs of one report."""

    def __init__(self, root: Path, resnet: Path | None) -> None:
        self.root = root
        self.ctx = bs.load_root(root, "VGG-16")
        self.paper = {num(r["r_star"]): r for r in read_csv(root / "paper" / "table.csv") if r["criterion"] == "dropout_scratch_sr"}
        self.var = read_csv(root / "sgr_variants" / "table.csv")
        self.aurc = {r["criterion"]: r for r in read_csv(root / "shift" / "id_aurc.csv")}
        self.cost = read_csv(root / "cost" / "table.csv")
        self.res_e3 = [r for r in read_csv(resnet / "matched" / "e3.csv") if r.get("criterion")] if resnet else []
        self.e3 = self.ctx["e3"]
        im = root / "imagenet"
        self.imt = read_csv(im / "table.csv") if (im / "table.csv").exists() else []
        self.ims = {(r["model"], r["task"], r["criterion"]): r for r in read_csv(im / "summary.csv")} if (im / "summary.csv").exists() else {}
        self.imc = read_csv(im / "curves.csv") if (im / "curves.csv").exists() else []
        self.imi = {(r["model"], r["task"], r["criterion"], r["rule"], round(num(r["r_star"]), 4)): r for r in self.imt}

    @property
    def has_im(self) -> bool:
        """Whether the ImageNet csv files exist."""
        return bool(self.imt and self.ims and self.imc)

    def imv(self, model: str, task: str, crit: str, rule: str, r: float, col: str) -> float:
        """Value of ``results/imagenet/table.csv``."""
        return num(self.imi.get((model, task, crit, rule, round(r, 4)), {}).get(col))

    def im_rs(self, model: str, task: str) -> list[float]:
        """The r* rows of the paper's table."""
        return [row[0] for row in PAPER_TABLES[model, task]]

    def e3v(self, crit: str, r: float, col: str = "own_cov_mean", res: bool = False) -> float:
        """Value of the matched table (VGG, or ResNet-18 with ``res``)."""
        return num(row_at(self.res_e3 if res else self.e3, r, criterion=crit).get(col))

    def delta01(self, r: float) -> float:
        """Test coverage of SGR with delta 0.1 on the paper model."""
        return num(row_at(self.var, r, criterion="dropout_scratch_sr", variant="sgr_delta_0.1").get("test_cov_mean"))

    def cost_cov(self, fam: str, crit: str, size: int, r: float = 0.01) -> float:
        """Coverage of the cost sweep."""
        for x in self.cost:
            if x["family"] == fam and x["criterion"] == crit and int(num(x["size"])) == size and abs(num(x["r_star"]) - r) < 1e-9:
                return num(x["cov_mean"])
        return float("nan")


# ------------------------------------------------------------------------------------------------------ figures
def fig_repro(d: Data) -> bytes:
    """Coverage vs r*: paper, ours at the paper's risk, SGR delta 0.001 and 0.1."""
    rs = sorted(d.paper)
    fig, (ax,) = bs.panels(1, height=3.8, width=8.5)
    series = [
        ("Paper (Table 1, test coverage)", [num(d.paper[r]["paper_test_cov"]) for r in rs], PAPER_COLOR, "o-"),
        ("Ours, cut at the paper's risk", [num(d.paper[r]["cov_at_paper_mean"]) for r in rs], ORANGE, "s--"),
        ("Ours, SGR delta = 0.1", [d.delta01(r) for r in rs], PURPLE, "^-"),
        ("Ours, SGR delta = 0.001", [num(d.paper[r]["test_cov_mean"]) for r in rs], ENS_COLOR, "D-"),
    ]
    for lab, ys, c, st in series:
        ax.plot(rs, ys, st, color=c, label=lab, lw=1.6, ms=5)
    bs.style_ax(ax, "target risk r*", "test coverage")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


def fig_methods(d: Data) -> bytes:
    """AURC and AUGRC of the key methods."""
    fig, axes = bs.panels(2, height=3.0)
    labs = [name(c) for c in KEY_METHODS]
    for ax, col, ttl in zip(axes, ["AURC x1000_mean", "AUGRC x1000_mean"], ["AURC (x1000, lower is better)", "AUGRC (x1000, lower is better)"], strict=True):
        vals = [num(d.aurc.get(c, {}).get(col)) for c in KEY_METHODS]
        cols = [ENS_COLOR if c.startswith("ens") else bs.GREY for c in KEY_METHODS]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)), labs, fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        bs.style_ax(ax, ttl)
    axes[1].set_yticklabels([])
    bs.refresh_ticks(fig)
    return bs.finish(fig)


def fig_rules(d: Data) -> bytes:
    """Coverage per rule for the deep ensemble at three targets; violations written next to unguaranteed rules."""
    order = ["sgr", "LTT FS", "FS split", "LTT Bonf", "emp", "Conf LAC", "Chow raw"]
    fig, axes = bs.panels(3, height=3.3)
    for ax, r in zip(axes, KEY_R, strict=True):
        for i, m in enumerate(order):
            cov = bs.opt_val(d.ctx, r, "ens_maxprob", f"{m} cov_mean")
            viol = bs.opt_val(d.ctx, r, "ens_maxprob", f"{m} viol_mean")
            ok = m in bs.GUARANTEED
            ax.barh(i, cov, color="#1b7f5c" if ok else bs.GREY)
            ax.text(cov, i, f" {pct(viol, 0)} violate" if not ok else " ok", va="center", fontsize=7)
        ax.set_yticks(range(len(order)), [RULE_LABELS[m] for m in order] if r == KEY_R[0] else [], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0.4, 1.12)
        bs.style_ax(ax, "coverage", None, f"r* = {r:g}")
    bs.refresh_ticks(fig)
    return bs.finish(fig, (0, 0, 1, 1))


def fig_improve(d: Data) -> bytes:
    """Coverage vs r*: paper, ensemble + SGR and softmax + SGR for both architectures."""
    rs = [0.01, 0.02, 0.03, 0.05]
    fig, axes = bs.panels(2, height=3.5, width=10)
    for ax, res, ttl in zip(axes, [False, True], ["VGG-16", "ResNet-18"], strict=True):
        if res and not d.res_e3:
            ax.set_visible(False)
            continue
        ax.plot(rs, [num(d.paper[r]["paper_test_cov"]) for r in rs], "o--", color=PAPER_COLOR, label="Paper (VGG-16)", lw=1.6)
        ax.plot(rs, [d.e3v("sr_base", r, res=res) for r in rs], "s-", color=SOFT_COLOR, label="Softmax + SGR", lw=1.6)
        ens = np.array([d.e3v("ens_maxprob", r, res=res) for r in rs])
        std = np.array([d.e3v("ens_maxprob", r, "own_cov_std", res=res) for r in rs])
        ax.plot(rs, ens, "D-", color=ENS_COLOR, label="Deep ensemble + SGR", lw=2.2)
        ax.fill_between(rs, ens - std, ens + std, color=ENS_COLOR, alpha=0.12)
        bs.style_ax(ax, "target risk r*", "test coverage", ttl)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


def fig_cost(d: Data) -> bytes:
    """Coverage at r* = 0.01 against forward passes."""
    fig, (ax,) = bs.panels(1, height=3.4, width=8.0)
    ens = [(int(num(x["passes"])), num(x["cov_mean"])) for x in d.cost if x["family"] == "ensemble" and x["criterion"] == "maxprob" and abs(num(x["r_star"]) - 0.01) < 1e-9]
    mc = [(int(num(x["passes"])), num(x["cov_mean"])) for x in d.cost if x["family"] == "mc_dropout" and x["criterion"] == "maxprob" and abs(num(x["r_star"]) - 0.01) < 1e-9]
    for pts_, lab, c, st in [(sorted(ens), "Deep ensemble (members)", ENS_COLOR, "D-"), (sorted(mc), "MC dropout (passes)", ORANGE, "o--")]:
        ax.plot(*zip(*pts_, strict=True), st, color=c, label=lab, lw=1.6)
    ax.axhline(num(d.paper[0.01]["paper_test_cov"]), color=PAPER_COLOR, ls=":", label="Paper")
    ax.set_xscale("log")
    bs.style_ax(ax, "forward passes per image", "coverage at r* = 0.01")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


def fig_ood(d: Data) -> bytes:
    """AUROC of the key methods for detecting SVHN and CIFAR-100 images."""
    ood = {k: {r["criterion"]: r for r in read_csv(d.root / "shift" / f"ood_{k}.csv")} for k in ("svhn", "cifar100")}
    fig, (ax,) = bs.panels(1, height=3.0, width=8.5)
    w = 0.38
    for j, (k, lab, hatch) in enumerate([("svhn", "SVHN", ""), ("cifar100", "CIFAR-100", "//")]):
        vals = [num(ood[k].get(c, {}).get("AUROC_mean")) for c in KEY_METHODS]
        ax.bar(np.arange(len(vals)) + (j - 0.5) * w, vals, w, color=[ENS_COLOR if c.startswith("ens") else bs.GREY for c in KEY_METHODS], hatch=hatch, edgecolor="white", label=lab)
    ax.set_xticks(range(len(KEY_METHODS)), [name(c).replace(", ", "\n") for c in KEY_METHODS], fontsize=7)
    ax.set_ylim(0.6, 1.0)
    bs.style_ax(ax, None, "AUROC (OOD detection)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", title="OOD set (plain / hatched)")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


# ----------------------------------------------------------------------------------------------- ImageNet figures
IM_MODELS = [("vgg16", "VGG-16"), ("resnet50", "ResNet-50")]
IM_TASKS = [("top1", "top-1"), ("top5", "top-5")]


def fig_im_curves(d: Data) -> bytes:
    """Risk-coverage curves (Fig. 2c style) with the paper's test points."""
    fig, axes = bs.panels(2, height=3.6, width=10)
    styles = {"sr": ("SR", BLUE), "mc_variance": ("MC variance", RED), "top5_mass": ("1 - top-5 mass", SOFT_COLOR)}
    for ax, (model, mname) in zip(axes, IM_MODELS, strict=True):
        for (task, tname), ls in zip(IM_TASKS, ("--", "-"), strict=True):
            for c, (lab, col) in styles.items():
                pts_ = [(num(x["coverage"]), num(x["risk"])) for x in d.imc if (x["model"], x["task"], x["criterion"]) == (model, task, c)]
                if pts_:
                    ax.plot(*zip(*pts_, strict=True), ls=ls, color=col, lw=1.4, label=f"{lab}, {tname}")
            p = np.array([(row[4], row[3]) for row in PAPER_TABLES[model, task]])
            ax.plot(p[:, 0], p[:, 1], "s" if task == "top1" else "o", color=PAPER_COLOR, ms=4, mfc="none", label=f"paper, {tname}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.35)
        bs.style_ax(ax, "coverage", "risk" if model == "vgg16" else None, mname)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


def fig_im_cov(d: Data) -> bytes:
    """Test coverage against r*: paper, SGR (delta 0.001 and 0.1), empirical threshold."""
    fig, axes = bs.panels(4, height=3.2, width=12)
    for ax, (model, task, tname) in zip(axes, [(m, t, tn) for m, _ in IM_MODELS for t, tn in IM_TASKS], strict=True):
        rs = d.im_rs(model, task)
        ax.plot(rs, [row[4] for row in PAPER_TABLES[model, task]], "o-", color=PAPER_COLOR, lw=1.4, ms=4, label="paper")
        for crit, rule, lab, col, st in [
            ("sr", "sgr", "SR, SGR 0.001", BLUE, "D-"),
            ("sr", "sgr_delta_0.1", "SR, SGR 0.1", PURPLE, "^--"),
            ("sr", "emp", "SR, empirical", bs.GREY, "s:"),
            ("top5_mass", "sgr", "1 - top-5 mass, SGR 0.001", SOFT_COLOR, "D-"),
        ]:
            if (model, task, crit, rule, round(rs[0], 4)) in d.imi:
                ax.plot(rs, [d.imv(model, task, crit, rule, r, "test_cov_mean") for r in rs], st, color=col, lw=1.3, ms=3.5, label=lab)
        bs.style_ax(ax, "target risk r*", "test coverage" if model == "vgg16" and task == "top1" else None, f"Table {PAPER_TABLE_NUMBER[model, task]}: {dict(IM_MODELS)[model]} {tname}")
    # The top-5 mass series only exists in the top-5 panels, so the legend collects the series of all panels.
    handles = {lab: h for ax in axes for h, lab in zip(*ax.get_legend_handles_labels(), strict=True)}
    axes[0].legend(handles.values(), handles.keys(), frameon=False, fontsize=7, loc="lower right")
    bs.refresh_ticks(fig)
    return bs.finish(fig)


# ------------------------------------------------------------------------------------------------------- tables
def diff_table() -> str:
    """Implementation differences to the paper, grouped."""
    rows = [
        ["<b>SGR and protocol</b>", "", "", ""],
        ["Returned threshold", "last tested iterate (Algorithm 1)", "<code>SGRSelector</code> returns the last certified iterate", "abstains where the last tested bound is above r*; mean coverage unchanged (section 4)"],
        ["Largest score", "not specified", "never tested; tied scores are accepted together", "none seen"],
        ["Bound", "delta / ceil(log2 m) per step", "same, exact Clopper-Pearson", "the printed bounds are not reproducible (section 6)"],
        ["Splits", "one split", "CIFAR: 5 seeds x 10 splits; ImageNet: 100 halvings", "means with spread instead of single values"],
        ["Framework", "Keras", "PyTorch, probly selectors and metrics", "-"],
        ["<b>CIFAR-10</b>", "", "", ""],
        ["Model", "VGG-16 with dropout", "own rebuild (training details: colleague), accuracy 93.59% (paper 93.54%)", "-"],
        ["Other models, CIFAR-100", "Table 2 (CIFAR-100)", "ResNet-18 and further methods are ours; CIFAR-100 not run", "no Table 2 comparison"],
        ["<b>ImageNet</b>", "", "", ""],
        ["Weights", "not stated", "torchvision IMAGENET1K_V1", "our ResNet-50 is likely not the paper's model (section 5)"],
        ["Preprocessing", "not stated", "resize 256, center crop 224", "Keras preprocessing not tried"],
        ["Top-5 score", "SR", "SR as in the paper, plus 1 - top-5 mass", "top-5 mass ranks better (section 5)"],
        ["MC dropout", "last FC layer", "both classifier dropouts, cached features, 100 passes", "-"],
        ["Data", "ILSVRC2012 validation", "Hugging Face mirror of the validation set", "-"],
    ]
    return table(["Item", "Paper", "Ours", "Effect"], rows).replace("<table>", '<table class="txt">', 1)


def im_acc_table(d: Data) -> str:
    """Accuracy, AURC and line-up per ImageNet model and loss (SR)."""
    rows = []
    for model, mname in IM_MODELS:
        for task, tname in IM_TASKS:
            sm = d.ims.get((model, task, "sr"), {})
            rs = d.im_rs(model, task)
            dr = [d.imv(model, task, "sr", "sgr", r, "risk_at_paper_mean") - row[3] for r, row in zip(rs, PAPER_TABLES[model, task], strict=True)]
            dc = [d.imv(model, task, "sr", "sgr", r, "cov_at_paper_mean") - row[4] for r, row in zip(rs, PAPER_TABLES[model, task], strict=True)]
            rows.append(
                [
                    f"{PAPER_TABLE_NUMBER[model, task]}: {mname} {tname}",
                    pct(1 - num(sm.get("full_risk"))),
                    f"{num(sm.get('aurc')):.4f}",
                    f"{np.mean(np.abs(dr)):.4f}",
                    f"{pts(min(dc))} to {pts(max(dc))}",
                ]
            )
    return table(["Paper table", "Accuracy", "AURC (SR)", "Mean |risk difference|", "Coverage difference at the paper's risk (points)"], rows)


def im_sgr_table(d: Data) -> str:
    """SGR vs the paper's coverage and violations, per table."""
    rows = []
    for model, mname in IM_MODELS:
        for task, tname in IM_TASKS:
            rs = d.im_rs(model, task)
            pc = PAPER_TABLES[model, task]
            viol = lambda rule: max(d.imv(model, task, "sr", rule, r, "viol_mean") for r in rs)  # noqa: E731
            rows.append(
                [
                    f"{PAPER_TABLE_NUMBER[model, task]}: {mname} {tname}",
                    f"{pct(pc[0][4])} / {pct(pc[-1][4])}",
                    f"{pct(d.imv(model, task, 'sr', 'sgr', rs[0], 'test_cov_mean'))} / {pct(d.imv(model, task, 'sr', 'sgr', rs[-1], 'test_cov_mean'))}",
                    f"{pct(d.imv(model, task, 'sr', 'sgr_delta_0.1', rs[0], 'test_cov_mean'))} / {pct(d.imv(model, task, 'sr', 'sgr_delta_0.1', rs[-1], 'test_cov_mean'))}",
                    pct(viol("sgr"), 0),
                    pct(viol("emp"), 0),
                ]
            )
    return table(
        ["Paper table", "Paper coverage (smallest / largest r*)", "SGR 0.001", "SGR 0.1", "Violating splits, SGR (worst r*)", "Violating splits, empirical (worst r*)"],
        rows,
    )


def im_fig2c_table(d: Data) -> str:
    """Risk at 60% coverage (Fig. 2c)."""
    rows = []
    for model, mname in IM_MODELS:
        for task, tname in IM_TASKS:
            for c, lab in [("sr", "SR"), ("mc_variance", "MC variance"), ("mc_maxprob", "MC max-prob"), ("top5_mass", "1 - top-5 mass")]:
                sm = d.ims.get((model, task, c))
                if sm:
                    rows.append([f"{mname} {tname}", lab, f"{num(sm['risk_at_60']):.3f}", f"{num(sm['aurc']):.4f}"])
    return table(["Model, loss", "Score", "Risk at 60% coverage", "AURC"], rows)


def im_top5_table(d: Data) -> str:
    """SGR coverage of SR vs top-5 mass for the top-5 loss."""
    rows = []
    for model, mname in IM_MODELS:
        for r in (0.01, 0.02, 0.05):
            rows.append([mname, f"{r:g}", pct(d.imv(model, "top5", "sr", "sgr", r, "test_cov_mean")), pct(d.imv(model, "top5", "top5_mass", "sgr", r, "test_cov_mean"))])
    return table(["Model", "r*", "SGR coverage, SR", "SGR coverage, 1 - top-5 mass"], rows)


def bound_checks(with_im: bool) -> list[tuple[str, list[dict[str, float]]]]:
    """Bound check rows per paper table (Table 1, and Tables 3-6 with ``with_im``)."""
    out = [("1: CIFAR-10 VGG-16", paper_check_rows())]
    if with_im:
        out += [(f"{PAPER_TABLE_NUMBER[m, t]}: {mn} {tn}", paper_bound_check(m, t)) for m, mn in IM_MODELS for t, tn in IM_TASKS]
    return out


def bound_table(d: Data, with_im: bool) -> str:
    """B* recomputed from the printed train numbers, one row per paper table, the full table folded."""
    rng = lambda v, f: f"{min(v):{f}} to {max(v):{f}}"  # noqa: E731
    short, full = [], []
    for lab, xs in bound_checks(with_im):
        short.append(
            [
                lab,
                str(len(xs)),
                rng([x["bound_split"] - x["paper_bound"] for x in xs], ".4f"),
                rng([x["bound_full"] - x["paper_bound"] for x in xs], ".4f"),
                rng([x["implied_delta"] for x in xs], ".3f"),
            ]
        )
        full += [[lab, f"{x['r']:.2f}", f"{x['paper_bound']:.4f}", f"{x['bound_split']:.4f}", f"{x['bound_full']:.4f}", f"{x['implied_delta']:.3f}"] for x in xs]
    t_short = table(["Paper table", "Rows", "B* - printed, delta/ceil(log2 m)", "B* - printed, delta", "Implied delta"], short)
    t_full = table(["Paper table", "r*", "Printed bound", "B*, delta/ceil(log2 m)", "B*, delta", "Implied delta"], full)
    return f"{t_short}<details><summary>Bound per row</summary>{t_full}</details>"


def train_risk_table(d: Data) -> str:
    """Train (selection-half) risk where SGR stops, per delta, next to the paper's."""
    rows = []
    for model, mname in IM_MODELS:
        for task, tname in IM_TASKS:
            for r, tr, *_ in PAPER_TABLES[model, task]:
                g = lambda rule: f"{d.imv(model, task, 'sr', rule, r, 'sel_risk_mean'):.4f}"  # noqa: E731
                rows.append([f"{PAPER_TABLE_NUMBER[model, task]}: {mname} {tname}", f"{r:.2f}", f"{tr:.4f}", g("sgr"), g("sgr_delta_0.01"), g("sgr_delta_0.1"), g("emp")])
    return table(["Paper table", "r*", "Paper", "SGR 0.001", "SGR 0.01", "SGR 0.1", "Empirical"], rows)


def methods_table() -> str:
    """Short table of the methods."""
    rows = [
        ["Softmax (base model)", "Standard VGG-16, confidence = highest softmax probability", "1"],
        ["Paper model, softmax", "VGG-16 with dropout trained from scratch, softmax score (the paper's setup)", "1"],
        ["MC dropout", "Base model with dropout layers, fine-tuned; dropout kept on at test time, 100 passes averaged", "100"],
        ["Deep ensemble", "5 independently trained models, predictions averaged", "5"],
        ["Ensemble, epistemic", "Deep ensemble, score = disagreement between members", "5"],
    ]
    return table(["Name", "What it is", "Forward passes"], rows).replace("<table>", '<table class="txt">', 1)


def rules_table() -> str:
    """Short table of the threshold rules."""
    rows = [
        [RULE_LABELS["sgr"], "Binary search over thresholds with a Clopper-Pearson-style bound (the paper)", "yes"],
        [RULE_LABELS["LTT FS"], "Test thresholds from strict to loose, stop at the first failure", "yes"],
        [RULE_LABELS["FS split"], "Same, but search and test on separate halves of the selection set", "yes"],
        [RULE_LABELS["LTT Bonf"], "Test each of 100 candidate thresholds at delta / 100, keep the certified one with the largest coverage", "yes"],
        [RULE_LABELS["emp"], "Largest coverage whose observed risk is below r*", "no"],
        [RULE_LABELS["Conf LAC"], "Conformal prediction sets, accept when the set is a single class", "no"],
        [RULE_LABELS["Chow raw"], "Accept when softmax confidence is at least 1 - r*", "no"],
    ]
    return table(["Name", "How the threshold is chosen", "Guarantee"], rows).replace("<table>", '<table class="txt">', 1)


def repro_table(d: Data) -> str:
    """Reproduction table at the key targets."""
    rows = [
        [
            f"{r:g}",
            pct(num(d.paper[r]["paper_test_cov"])),
            pct(num(d.paper[r]["cov_at_paper_mean"])),
            pct(num(d.paper[r]["test_cov_mean"])),
            pct(num(d.paper[r]["certified_mean"], ), 0),
        ]
        for r in KEY_R
    ]
    return table(["r*", "Paper", "Ours, at the paper's risk", "Ours, SGR delta = 0.001", "Certified splits"], rows)


def method_cmp_table(d: Data) -> str:
    """Short comparison table of the key methods."""
    rows = []
    for c in KEY_METHODS:
        a = d.aurc.get(c, {})
        rows.append(
            [
                name(c),
                f"{num(a.get('AURC x1000_mean')):.2f}",
                f"{num(a.get('AUGRC x1000_mean')):.2f}",
                pct(d.e3v(c, 0.02)),
                pct(d.e3v(c, 0.01, "own_cert"), 0),
            ]
        )
    return table(["Method", "AURC x1000", "AUGRC x1000", "SGR coverage, r* 0.02", "Certified, r* 0.01"], rows)


def rules_cmp_table(d: Data) -> str:
    """Short table of rules at r* = 0.02 for the ensemble."""
    rows = []
    for m in ["sgr", "LTT FS", "FS split", "LTT Bonf", "emp", "Conf LAC", "Chow raw"]:
        cov = bs.opt_val(d.ctx, 0.02, "ens_maxprob", f"{m} cov_mean")
        viol = bs.opt_val(d.ctx, 0.02, "ens_maxprob", f"{m} viol_mean")
        flag = '<span class="g">yes</span>' if m in bs.GUARANTEED else '<span class="n">no</span>'
        rows.append([RULE_LABELS[m], flag, pct(cov), pct(viol, 0)])
    return table(["Rule (deep ensemble, r* = 0.02)", "Guarantee", "Coverage", "Violating splits"], rows)


def improve_table(d: Data) -> str:
    """Improvement table."""
    rows = []
    for r in KEY_R:
        rows.append(
            [
                f"{r:g}",
                pct(num(d.paper[r]["paper_test_cov"])),
                pct(d.e3v("sr_base", r)),
                pct(d.e3v("ens_maxprob", r)),
                pct(d.e3v("sr_base", r, res=True)) if d.res_e3 else "n/a",
                pct(d.e3v("ens_maxprob", r, res=True)) if d.res_e3 else "n/a",
            ]
        )
    return table(["r*", "Paper", "Softmax + SGR", "Ensemble + SGR", "ResNet-18 softmax + SGR", "ResNet-18 ensemble + SGR"], rows)


CAL_NAMES = {"mc": "MC dropout", "ens": "Deep ensemble", "finetune": "Fine-tuned base model", "swag": "SWAG"}


def other_tables(d: Data) -> tuple[str, str]:
    """Folded calibration and OOD tables."""
    cal = read_csv(d.root / "calibration" / "clean_calibration.csv")[:6]
    t_cal = table(
        ["Source", "Accuracy", "ECE raw", "ECE temp. scaled"],
        [[name(r["source"]) if r["source"] in CRITERIA else CAL_NAMES.get(r["source"], r["source"]), f"{num(r['accuracy_mean']):.3f}", f"{num(r['ECE raw_mean']):.3f}", f"{num(r['ECE TS_mean']):.3f}"] for r in cal],
    )
    ood = {k: {r["criterion"]: r for r in read_csv(d.root / "shift" / f"ood_{k}.csv")} for k in ("svhn", "cifar100")}
    t_ood = table(
        ["Method", "AUROC vs SVHN", "AUROC vs CIFAR-100"],
        [[name(c), f"{num(ood['svhn'].get(c, {}).get('AUROC_mean')):.3f}", f"{num(ood['cifar100'].get(c, {}).get('AUROC_mean')):.3f}"] for c in KEY_METHODS],
    )
    return (
        f"<details><summary>Calibration table</summary>{t_cal}</details>",
        f"<details><summary>OOD table</summary>{t_ood}</details>",
    )


def png(path: Path, alt: str) -> str:
    """Embed an existing PNG, or a note."""
    return img_tag(path.read_bytes(), alt) if path.exists() else f"<p><em>{alt} not available.</em></p>"


# --------------------------------------------------------------------------------------------------------- numbers
def im_numbers(d: Data) -> dict[str, str]:
    """Numbers of the ImageNet sections and of the bound check (empty-safe)."""
    cif = [x["implied_delta"] for x in paper_check_rows()]
    out = {"cif_d_lo": f"{min(cif):.2f}", "cif_d_hi": f"{max(cif):.2f}"}
    if not d.has_im:
        return out
    imp = [x["implied_delta"] for m, _ in IM_MODELS for t, _ in IM_TASKS for x in paper_bound_check(m, t)]
    imp = [x for x in imp if not math.isnan(x)]
    out["n_pairs_im"] = "100"
    out |= {"im_d_lo": f"{min(imp):.2f}", "im_d_hi": f"{max(imp):.2f}"}
    sg = lambda rule, m, t: [d.imv(m, t, "sr", rule, r, "viol_mean") for r in d.im_rs(m, t)]  # noqa: E731
    allr = [(m, t) for m, _ in IM_MODELS for t, _ in IM_TASKS]
    out["im_viol_sgr"] = pct(max(v for m, t in allr for v in sg("sgr", m, t)), 0)
    emp = [max(sg("emp", m, t)) for m, t in allr]
    out["im_viol_emp_lo"], out["im_viol_emp_hi"] = pct(min(emp), 0), pct(max(emp), 0)
    dc = [d.imv("resnet50", t, "sr", "sgr", r, "cov_at_paper_mean") - row[4] for t, _ in IM_TASKS for r, row in zip(d.im_rs("resnet50", t), PAPER_TABLES["resnet50", t], strict=True)]
    out["res_cd_lo"], out["res_cd_hi"] = pts(min(dc)), pts(max(dc))
    # SGR (delta 0.001 and 0.1) against the paper's test coverage, per row of Tables 3-6.
    cmp = [
        (m, t, r, d.imv(m, t, "sr", "sgr", r, "test_cov_mean"), d.imv(m, t, "sr", "sgr_delta_0.1", r, "test_cov_mean"), row[4])
        for m, t in allr
        for r, row in zip(d.im_rs(m, t), PAPER_TABLES[m, t], strict=True)
    ]
    less = [x for x in cmp if x[3] < x[5]]
    out["im_rows"], out["im_more"] = str(len(cmp)), str(len(cmp) - len(less))
    out["im_less_text"] = "It never accepts less."
    if less:
        m, t, r, s, _, pc = max(less, key=lambda x: x[5] - x[3])
        d01 = max(max(x[5] - x[4], 0.0) for x in less)
        out["im_less_text"] = (
            f"It accepts less in the other {len(less)} rows, all at r* = {max(x[2] for x in less):g} or below, by up to "
            f"{pts(pc - s)} points (Table {PAPER_TABLE_NUMBER[m, t]} at r* = {r:g}: {pct(s)} against {pct(pc)}); there, "
            f"SGR at delta = 0.1 comes within {pts(d01)} points of the paper."
        )
    sd = [d.imv(m, t, "sr", "sgr", r, "test_risk_std") for m, t, r, *_ in cmp]
    out["im_sd_lo"], out["im_sd_hi"] = f"{min(sd):.4f}", f"{max(sd):.4f}"
    # The paper's test risk and coverage bound its full-coverage error from below (rejected points correct) and above.
    p5 = PAPER_TABLES["resnet50", "top1"]
    out["pacc_lo"] = pct(1 - min(row[3] * row[4] + 1 - row[4] for row in p5))
    out["pacc_hi"] = pct(1 - max(row[3] * row[4] for row in p5))
    for model, key in [("vgg16", "vgg"), ("resnet50", "res")]:
        for task, k2 in [("top1", "1"), ("top5", "5")]:
            rs = d.im_rs(model, task)
            dr = [abs(d.imv(model, task, "sr", "sgr", r, "risk_at_paper_mean") - row[3]) for r, row in zip(rs, PAPER_TABLES[model, task], strict=True)]
            out[f"lu_{key}{k2}"] = f"{np.mean(dr):.4f}"
            out[f"acc_{key}{k2}"] = pct(1 - num(d.ims[model, task, "sr"]["full_risk"]))
    for k, key in [("sr", "sr"), ("mc_variance", "mcv"), ("mc_maxprob", "mcm")]:
        out[f"r60_{key}1"] = f"{num(d.ims['vgg16', 'top1', k]['risk_at_60']):.3f}"
    out["r60_sr5"] = f"{num(d.ims['vgg16', 'top5', 'sr']['risk_at_60']):.3f}"
    out["r60_mcv5"] = f"{num(d.ims['vgg16', 'top5', 'mc_variance']['risk_at_60']):.3f}"
    for model, key in [("vgg16", "vgg"), ("resnet50", "res")]:
        for r, k in [(0.01, "01"), (0.02, "02")]:
            out[f"t5sr_{key}{k}"] = pct(d.imv(model, "top5", "sr", "sgr", r, "test_cov_mean"))
            out[f"t5m_{key}{k}"] = pct(d.imv(model, "top5", "top5_mass", "sgr", r, "test_cov_mean"))
    rows = [(d.imv(m, t, "sr", "sgr_delta_0.1", r, "sel_risk_mean"), row[1]) for m, t in allr for r, row in zip(d.im_rs(m, t), PAPER_TABLES[m, t], strict=True)]
    out["tr_ge"], out["tr_n"] = str(sum(p >= o - 1e-9 for o, p in rows)), str(len(rows))
    same = sum(abs(a[1] - b[1]) <= 1.05e-4 for t in ("top1", "top5") for a, b in zip(PAPER_TABLES["vgg16", t], PAPER_TABLES["resnet50", t], strict=True))
    out["tr_same"], out["tr_total"] = str(same), str(sum(len(PAPER_TABLES["vgg16", t]) for t in ("top1", "top5")))
    sgr_v = [x for x in d.var if x["criterion"] == "dropout_scratch_sr" and abs(num(x["r_star"]) - 0.01) < 1e-9]
    for v, k in [("sgr", "cert"), ("sgr_last_tested", "last")]:
        row = next((x for x in sgr_v if x["variant"] == v), {})
        out[f"c01_{k}"] = pct(num(row.get("test_cov_mean")))
    same_c01 = out["c01_last"] == out["c01_cert"]
    out["c01_cmp"] = f"the same ({out['c01_cert']} for both)" if same_c01 else f"{out['c01_last']} against {out['c01_cert']}"
    return out


def numbers(d: Data) -> dict[str, str]:
    """All numbers quoted in the prose."""
    rs = sorted(d.paper)
    gains = [num(d.paper[r]["cov_at_paper_mean"]) - num(d.paper[r]["paper_test_cov"]) for r in rs]
    gaps = [num(d.paper[r]["paper_test_cov"]) - num(d.paper[r]["test_cov_mean"]) for r in rs if r >= 0.02]
    rows_all = [(c, v) for rr in d.ctx["opt"].values() for c, v in rr.items()]
    viol = max((num(v.get("sgr viol_mean")) for c, v in rows_all if c in [*KEY_METHODS, "sr_dropout"]), default=float("nan"))
    viol_all = max((num(v.get("sgr viol_mean")) for _, v in rows_all), default=float("nan"))
    ac = lambda c, col: f"{num(d.aurc.get(c, {}).get(col)):.2f}"  # noqa: E731
    rule_viol = lambda m: [bs.opt_val(d.ctx, r, "ens_maxprob", f"{m} viol_mean") for r in KEY_R]  # noqa: E731
    cov_rule = lambda m: [bs.opt_val(d.ctx, r, "ens_maxprob", f"{m} cov_mean") for r in KEY_R]  # noqa: E731
    bonf_gap = [a - b for a, b in zip(cov_rule("sgr"), cov_rule("LTT Bonf"), strict=True)]
    cal = {r["source"]: r for r in read_csv(d.root / "calibration" / "clean_calibration.csv")}
    ood = {r["criterion"]: r for r in read_csv(d.root / "shift" / "ood_svhn.csv")}
    ft = [num(r["accuracy_mean"]) for r in read_csv(d.root / "shift" / "id_aurc.csv") if r["criterion"].startswith(("swa", "swag", "ddu", "vbll", "sngp"))]
    return {
        **im_numbers(d),
        "n_pairs": "50",
        "emp_viol_lo": pct(min(rule_viol("emp")), 0),
        "emp_viol_hi": pct(max(rule_viol("emp")), 0),
        "conf_viol_hi": pct(max(rule_viol("Conf LAC")), 0),
        "bonf_gap_lo": pts(min(bonf_gap)),
        "bonf_gap_hi": pts(max(bonf_gap)),
        "chow_005": pct(bs.opt_val(d.ctx, 0.05, "ens_maxprob", "Chow raw cov_mean")),
        "sgr_005": pct(bs.opt_val(d.ctx, 0.05, "ens_maxprob", "sgr cov_mean")),
        "mc_passes": str(MC_PASSES),
        "repro_gain_lo": pts(min(gains)),
        "repro_gain_hi": pts(max(gains)),
        "sgr_gap_lo": pts(min(gaps)),
        "sgr_gap_hi": pts(max(gaps)),
        "d01_002": pct(d.delta01(0.02)),
        "cert_001": pct(num(d.paper[0.01]["certified_mean"]), 0),
        "sgr_cov_001": pct(num(d.paper[0.01]["test_cov_mean"])),
        "sgr_viol_max": "none" if viol == 0 else f"at most {100 * viol:.0f}%",
        "sgr_viol_all": f"{100 * viol_all:.0f}%",
        "ens_cov_001": pct(d.e3v("ens_maxprob", 0.01)),
        "paper_cov_001": pct(num(d.paper[0.01]["paper_test_cov"])),
        "ens_cert_001": pct(d.e3v("ens_maxprob", 0.01, "own_cert"), 0),
        "soft_cov_001": pct(d.e3v("sr_base", 0.01)),
        "soft_cert_001": pct(d.e3v("sr_base", 0.01, "own_cert"), 0),
        "res_ens_001": pct(d.e3v("ens_maxprob", 0.01, res=True)),
        "res_soft_001": pct(d.e3v("sr_base", 0.01, res=True)),
        "a_ens": ac("ens_maxprob", "AUGRC x1000_mean"),
        "a_soft": ac("sr_base", "AUGRC x1000_mean"),
        "a_mc": ac("mc_maxprob", "AUGRC x1000_mean"),
        "a_drop": ac("sr_dropout", "AUGRC x1000_mean"),
        "acc_ft": f"{min(ft):.3f}-{max(ft):.3f}" if ft else "n/a",
        "acc_base": f"{num(d.aurc.get('sr_base', {}).get('accuracy_mean')):.4f}",
        "cost_e2": pct(d.cost_cov("ensemble", "maxprob", 2)),
        "cost_e5": pct(d.cost_cov("ensemble", "maxprob", 5)),
        "cost_mc": pct(d.cost_cov("mc_dropout", "maxprob", MC_PASSES)),
        "ece_raw": f"{num(cal.get('sr_base', {}).get('ECE raw_mean')):.3f}",
        "ece_ts": f"{num(cal.get('sr_base', {}).get('ECE TS_mean')):.3f}",
        "auroc_ens": f"{num(ood.get('ens_maxprob', {}).get('AUROC_mean')):.3f}",
        "auroc_base": f"{num(ood.get('sr_base', {}).get('AUROC_mean')):.3f}",
    }


def appendix(d: Data) -> str:
    """Raw-id appendix, folded."""
    blocks = [
        ("Paper reproduction (raw)", read_csv(d.root / "paper" / "table.csv")),
        ("AURC / AUGRC of all criteria (raw ids)", read_csv(d.root / "shift" / "id_aurc.csv")),
        ("Matched comparison, VGG-16 (raw ids)", d.e3),
        ("Matched comparison, ResNet-18 (raw ids)", d.res_e3),
        ("Cost sweep (raw)", d.cost),
        ("SGR variants (raw)", d.var),
    ]
    if d.has_im:
        blocks = [("ImageNet table (raw)", d.imt), ("ImageNet summary (raw)", list(d.ims.values())), *blocks]
    out = ["<p>Raw tables with the internal criterion ids. Open a block to see it.</p>"]
    out += [f"<details><summary>{t}</summary>{raw_table(rows)}</details>" for t, rows in blocks if rows]
    return "".join(out)


def build_html(d: Data) -> str:
    """Assemble the report."""
    n = numbers(d)
    S = rt.SECTIONS
    cal_t, ood_t = other_tables(d)
    shift_png = bs.fig_shift([d.ctx])
    im_note = "<p><em>ImageNet results not found (run imagenet_table.py first); this section is skipped.</em></p>"
    if d.has_im:
        im_repro = S["im_repro"].format(
            fig_curves=img_tag(fig_im_curves(d), "ImageNet risk-coverage curves"),
            fig_cov=img_tag(fig_im_cov(d), "ImageNet coverage"),
            acc_table=im_acc_table(d),
            sgr_table=im_sgr_table(d),
            fig2c_table=im_fig2c_table(d),
            top5_table=im_top5_table(d),
            **n,
        )
        train_t = f"<details><summary>Train risk per row</summary>{train_risk_table(d)}</details>"
    else:
        im_repro, train_t = im_note, ""
    parts = {
        "glance": S["glance"].format(**n) if d.has_im else S["glance_cifar"].format(**n),
        "setup": S["setup"].format(methods_table=methods_table(), rules_table=rules_table(), **n),
        "diff": S["diff"].format(table=diff_table(), **n),
        "repro": S["repro"].format(fig=img_tag(fig_repro(d), "reproduction"), table=repro_table(d), **n),
        "im_repro": im_repro,
        "bounds": S["bounds"].format(table=bound_table(d, d.has_im), train_table=train_t, **({**n} if d.has_im else {"im_d_lo": "n/a", "im_d_hi": "n/a", "tr_ge": "n/a", "tr_n": "n/a", "tr_same": "n/a", "tr_total": "n/a", **n})),
        "methods": S["methods"].format(fig=img_tag(fig_methods(d), "methods"), table=method_cmp_table(d), **n),
        "rules": S["rules"].format(fig=img_tag(fig_rules(d), "rules"), table=rules_cmp_table(d), **n),
        "improvement": S["improvement"].format(
            fig=img_tag(fig_improve(d), "ensemble + SGR"), table=improve_table(d), fig_cost=img_tag(fig_cost(d), "cost"), **n
        ),
        "other": S["other"].format(
            fig_cal=png(d.root / "calibration" / "ece_vs_severity.png", "Calibration"),
            fig_ood=img_tag(fig_ood(d), "OOD detection"),
            fig_shift=img_tag(shift_png, "shift") if shift_png else "",
            tab_cal=cal_t,
            tab_ood=ood_t,
            **n,
        ),
    }
    titles = [
        ("glance", "1. Summary"),
        ("setup", "2. Setup"),
        ("diff", "3. Implementation differences to the paper"),
        ("repro", "4. Replication, CIFAR-10 (Table 1)"),
        ("im_repro", "5. Replication, ImageNet (Tables 3-6, Fig. 2c)"),
        ("bounds", "6. The paper's printed bounds"),
        ("methods", "7. Uncertainty methods (CIFAR-10)"),
        ("rules", "8. Threshold rules (CIFAR-10)"),
        ("improvement", "9. Deep ensemble + SGR (CIFAR-10)"),
        ("other", "10. Other experiments"),
    ]
    body = "".join(f"<section><h2>{t}</h2>{parts[k]}</section>" for k, t in titles)
    body += f"<section><h2>11. Appendix</h2>{appendix(d)}</section>"
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{rt.TITLE}</title>'
        f"<style>{REPORT_CSS}</style></head><body><main><h1>{rt.TITLE}</h1><p class=\"sub\">{rt.SUBTITLE}</p>{body}</main></body></html>"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--resnet", type=Path, default=None, help="ResNet-18 results (default: sibling results_resnet18).")
    p.add_argument("--out", type=Path, default=None, help="Output HTML (default: <results>/report.html).")
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--chrome", default=None)
    return p.parse_args()


def main() -> None:
    """Build the report."""
    args = parse_args()
    bs.HAS_FIRA = setup_fonts()
    resnet = args.resnet or args.results.resolve().parent / "results_resnet18"
    d = Data(args.results, resnet if (resnet / "matched").is_dir() else None)
    out = args.out or args.results / "report.html"
    out.write_text(build_html(d) + "\n")
    print(f"Wrote {out}")
    plt.close("all")
    if args.pdf and bs.render_pdf(out, out.with_suffix(".pdf"), args.chrome):
        print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
