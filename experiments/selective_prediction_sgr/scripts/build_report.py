"""Build the readable HTML (and PDF) report of the selective-prediction experiment.

Order: at a glance, setup, reproduction of Geifman and El-Yaniv 2017 (Table 1), method comparison, threshold rules,
our improvement (deep ensemble + SGR), other experiments (calibration, OOD, shift), next steps, appendix with raw ids.
All numbers are read from the result csv files; the prose lives in ``report_text.py``. Tables in the main sections
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

KEY_R = [0.01, 0.02, 0.05]
KEY_METHODS = ["sr_base", "mc_maxprob", "ens_maxprob", "ens_epistemic", "dropout_scratch_sr"]
PAPER_COLOR = "#c62828"
ENS_COLOR = "#1e3a8a"
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
.g { color:#1b7f5c; font-weight:600; } .n { color:var(--mut); }
details { margin:10px 0; } summary { cursor:pointer; color:var(--mut); }
details table { font-size:12px; }
table.txt th, table.txt td { text-align:left; }
@media print { section { break-inside:avoid-page; } img { break-inside:avoid; } }
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
        ("Ours, cut at the paper's risk", [num(d.paper[r]["cov_at_paper_mean"]) for r in rs], "#e65100", "s--"),
        ("Ours, SGR delta = 0.1", [d.delta01(r) for r in rs], "#7e57c2", "^-"),
        ("Ours, SGR delta = 0.001 (valid)", [num(d.paper[r]["test_cov_mean"]) for r in rs], ENS_COLOR, "D-"),
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
    for pts_, lab, c, st in [(sorted(ens), "Deep ensemble (members)", ENS_COLOR, "D-"), (sorted(mc), "MC dropout (passes)", "#1e88e5", "o--")]:
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


# ------------------------------------------------------------------------------------------------------- tables
def methods_table() -> str:
    """Short table of the methods."""
    rows = [
        ["Softmax (base model)", "Standard VGG-16, confidence = highest softmax probability", "1"],
        ["Paper model, softmax", "VGG-16 with dropout trained from scratch, softmax score (the paper's setup)", "1"],
        ["MC dropout", "Dropout kept on at test time, 100 stochastic passes averaged", "100"],
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
    return table(["r*", "Paper", "Ours, at the paper's risk", "Ours, SGR (valid)", "Certified splits"], rows)


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
    for m in ["sgr", "LTT FS", "emp", "Conf LAC", "Chow raw"]:
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


def other_tables(d: Data) -> tuple[str, str]:
    """Folded calibration and OOD tables."""
    cal = read_csv(d.root / "calibration" / "clean_calibration.csv")[:6]
    t_cal = table(
        ["Source", "Accuracy", "ECE raw", "ECE temp. scaled"],
        [[name(r["source"]) if r["source"] in CRITERIA else r["source"], f"{num(r['accuracy_mean']):.3f}", f"{num(r['ECE raw_mean']):.3f}", f"{num(r['ECE TS_mean']):.3f}"] for r in cal],
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
def numbers(d: Data) -> dict[str, str]:
    """All numbers quoted in the prose."""
    rs = sorted(d.paper)
    gains = [num(d.paper[r]["cov_at_paper_mean"]) - num(d.paper[r]["paper_test_cov"]) for r in rs]
    gaps = [num(d.paper[r]["paper_test_cov"]) - num(d.paper[r]["test_cov_mean"]) for r in rs if r >= 0.02]
    rows_all = [(c, v) for rr in d.ctx["opt"].values() for c, v in rr.items()]
    viol = max((num(v.get("sgr viol_mean")) for c, v in rows_all if c in [*KEY_METHODS, "sr_dropout"]), default=float("nan"))
    viol_all = max((num(v.get("sgr viol_mean")) for _, v in rows_all), default=float("nan"))
    ac = lambda c, col: f"{num(d.aurc.get(c, {}).get(col)):.2f}"  # noqa: E731
    cal = {r["source"]: r for r in read_csv(d.root / "calibration" / "clean_calibration.csv")}
    ood = {r["criterion"]: r for r in read_csv(d.root / "shift" / "ood_svhn.csv")}
    ft = [num(r["accuracy_mean"]) for r in read_csv(d.root / "shift" / "id_aurc.csv") if r["criterion"].startswith(("swa", "swag", "ddu", "vbll", "sngp"))]
    return {
        "n_pairs": "50",
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
        "sgr_viol_max": f"{100 * viol:.0f}%",
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


def kpis(n: dict[str, str], d: Data) -> str:
    """Four KPI cards."""
    cards = [
        (n["ens_cov_001"], f"coverage of ensemble + SGR at r* 0.01 (paper: {n['paper_cov_001']})"),
        (n["ens_cert_001"], f"splits certified by the ensemble at r* 0.01 (softmax: {n['soft_cert_001']})"),
        (n["sgr_viol_max"], "split pairs where SGR broke the risk target"),
        (f"{n['res_ens_001']}", f"ResNet-18 ensemble coverage at r* 0.01 (softmax: {n['res_soft_001']})"),
    ]
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{v}</b><span>{lab}</span></div>' for v, lab in cards) + "</div>"


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
    out = ["<p>Raw tables with the internal criterion ids. Open a block to see it.</p>"]
    out += [f"<details><summary>{t}</summary>{raw_table(rows)}</details>" for t, rows in blocks if rows]
    return "".join(out)


def build_html(d: Data) -> str:
    """Assemble the report."""
    n = numbers(d)
    S = rt.SECTIONS
    cal_t, ood_t = other_tables(d)
    shift_png = bs.fig_shift([d.ctx])
    parts = {
        "glance": S["glance"].format(**n),
        "setup": S["setup"].format(methods_table=methods_table(), rules_table=rules_table(), **n),
        "repro": S["repro"].format(fig=img_tag(fig_repro(d), "reproduction"), table=repro_table(d), **n),
        "methods": S["methods"].format(fig=img_tag(fig_methods(d), "methods"), table=method_cmp_table(d), **n),
        "rules": S["rules"].format(fig=img_tag(fig_rules(d), "rules"), table=rules_cmp_table(d), **n),
        "improvement": S["improvement"].format(
            fig=img_tag(fig_improve(d), "improvement"), table=improve_table(d), fig_cost=img_tag(fig_cost(d), "cost"), **n
        ),
        "other": S["other"].format(
            fig_cal=png(d.root / "calibration" / "ece_vs_severity.png", "Calibration"),
            fig_ood=img_tag(fig_ood(d), "OOD detection"),
            fig_shift=img_tag(shift_png, "shift") if shift_png else "",
            tab_cal=cal_t,
            tab_ood=ood_t,
            **n,
        ),
        "next": S["next"],
    }
    titles = [
        ("glance", "1. At a glance"),
        ("setup", "2. Setup"),
        ("repro", "3. Reproducing the paper"),
        ("methods", "4. Comparing uncertainty methods"),
        ("rules", "5. Comparing threshold rules"),
        ("improvement", "6. Our improvement: deep ensemble + SGR"),
        ("other", "7. Other experiments"),
        ("next", "8. Next steps"),
    ]
    body = "".join(f"<section><h2>{t}</h2>{kpis(n, d) if k == 'glance' else ''}{parts[k]}</section>" for k, t in titles)
    body += f"<section><h2>9. Appendix</h2>{appendix(d)}</section>"
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
    out.write_text(build_html(d))
    print(f"Wrote {out}")
    plt.close("all")
    if args.pdf and bs.render_pdf(out, out.with_suffix(".pdf"), args.chrome):
        print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
