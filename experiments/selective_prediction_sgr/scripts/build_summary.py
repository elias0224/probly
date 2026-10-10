"""Slim, plot-heavy, single-file HTML summary of the selective-prediction experiment.

Each section is a heading, one computed takeaway sentence and a figure (PNG, base64-embedded), optionally with a tiny
table. One or two result roots are accepted (e.g. VGG-16 and ResNet-18); with two roots the figures get one panel per
root. Sections whose CSVs are missing are omitted with a note on stdout.

References: Geifman and El-Yaniv 2017 (SGR), Traub et al. 2024 (AUGRC), Angelopoulos et al. (Learn-then-Test).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import math
from pathlib import Path
import shutil
import subprocess
import warnings

from evaluate import THIN_FONT, setup_fonts
from evaluate_shift import CRITERIA, GUARANTEED_RULES, RULE_LABELS
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

NAN = float("nan")
DELTA = 0.001
GREY = "#9aa0a6"
MODES = ["emp", "sgr", "cov", "LTT Bonf", "LTT FS", "FS split", "Chow raw", "Chow TS", "Conf LAC", "Conf APS"]
GUARANTEED = GUARANTEED_RULES
CONFORMAL = ["Conf LAC", "Conf APS"]
MODE_COLORS = {"sgr": "#1b7f5c", "LTT Bonf": "#2e9e6f", "LTT FS": "#0b5d46", "FS split": "#5bbf8f"}
HERO_STYLE = {
    "sr_base": {"color": "#16a085", "ls": "-", "lw": 1.5, "label": "SR, base model"},
    "ens_maxprob": {"color": "#1e3a8a", "ls": "-", "lw": 2.4, "label": "Ensemble, max-prob"},
    "mc_maxprob": {"color": "#1e88e5", "ls": "--", "lw": 1.5, "label": "MC dropout, max-prob"},
    "dropout_scratch_sr": {"color": "#e65100", "ls": "-", "lw": 1.5, "label": "Dropout from scratch, SR"},
    "dropout_scratch_maxprob": {"color": "#ffb74d", "ls": "--", "lw": 1.5, "label": "Dropout from scratch, MC"},
}
for _c, _st in HERO_STYLE.items():  # one shared display-name mapping (evaluate_shift.CRITERIA)
    _st["label"] = CRITERIA[_c]["label"]
HERO_CRITERIA = ["sr_base", "ens_maxprob", "mc_maxprob", "dropout_scratch_sr", "dropout_scratch_maxprob"]
SHIFT_CRITERIA = ["sr_base", "ens_maxprob", "mc_maxprob"]
CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "contrast", "pixelate"]
HAS_FIRA = False


# ---------------------------------------------------------------------------------------------------------------- IO
def num(x: object) -> float:
    """Parse a float, mapping empty or invalid values to NaN."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return NAN


def truthy(x: object) -> bool:
    """Parse the boolean spellings used in the CSVs ("True", "1", "1.0")."""
    return str(x).strip().lower() in {"true", "1", "1.0"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dicts, skipping stray repeated header rows. Returns [] if the file is missing."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    first = next(iter(rows[0])) if rows else None
    return [r for r in rows if first is None or r.get(first) != first]


def subdir(root: Path, *names: str) -> Path:
    """Return the first existing subdirectory of ``root`` among ``names`` (else the first name)."""
    for n in names:
        if (root / n).is_dir():
            return root / n
    return root / names[0]


def load_options(root: Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load ``options/id_r*.csv`` as ``{r*: {criterion: row}}``."""
    out: dict[float, dict[str, dict[str, str]]] = {}
    for p in sorted((root / "options").glob("id_r*.csv")):
        try:
            r = float(p.stem.removeprefix("id_r"))
        except ValueError:
            continue
        out[r] = {row["criterion"]: row for row in read_csv(p)}
    return out


def load_root(root: Path, name: str) -> dict:
    """Load everything one result root offers."""
    e3 = [r for r in read_csv(root / "matched" / "e3.csv") if r.get("criterion")]
    e4 = [r for r in read_csv(root / "matched" / "e4.csv") if r.get("score")]
    shift = {}
    for c in CORRUPTIONS:
        rows = read_csv(root / "shift" / f"shift_{c}.csv")
        if rows:
            shift[c] = rows
    return {"name": name, "opt": load_options(root), "e3": e3, "e4": e4, "shift": shift}


def opt_val(ctx: dict, r: float, crit: str, col: str) -> float:
    """Value of an options column (e.g. ``sgr cov_mean``) or NaN."""
    return num(ctx["opt"].get(r, {}).get(crit, {}).get(col))


def near(r: float, rs: list[float]) -> float | None:
    """Return the member of ``rs`` equal to ``r`` up to rounding."""
    return next((x for x in rs if abs(x - r) < 1e-9), None)


# ---------------------------------------------------------------------------------------------------------- plotting
def crit_style(crit: str) -> dict:
    """Color, label and line style of a criterion, with grey fallbacks."""
    s = CRITERIA.get(crit, {})
    return {"label": s.get("label", crit), "color": s.get("color", GREY), "ls": s.get("ls", "-")}


def style_ax(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None) -> None:
    """Apply the shared axis style: semibold labels, thin ticks, light grid."""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="semibold")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if HAS_FIRA:
        for lab in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            lab.set_fontfamily(THIN_FONT)


def finish(fig: plt.Figure, rect: tuple[float, float, float, float] = (0, 0, 1, 1)) -> bytes:
    """Render a figure to PNG bytes (dpi 150) and close it."""
    fig.tight_layout(rect=rect)
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def panels(n: int, height: float = 3.4, width: float = 10.0) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create ``n`` side-by-side panels."""
    fig, axes = plt.subplots(1, n, figsize=(width, height), squeeze=False)
    return fig, list(axes[0])


def refresh_ticks(fig: plt.Figure) -> None:
    """Re-apply thin tick fonts after the ticks were finalized."""
    if HAS_FIRA:
        for ax in fig.axes:
            for lab in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                lab.set_fontfamily(THIN_FONT)


# ---------------------------------------------------------------------------------------------------------- sections
def family_color(crit: str) -> str:
    """Hero palette color of a criterion by family (base, ensemble, MC, dropout from scratch), else grey."""
    if crit.startswith("sr_base"):
        return HERO_STYLE["sr_base"]["color"]
    if crit.startswith("ens_"):
        return HERO_STYLE["ens_maxprob"]["color"]
    if crit.startswith("mc_"):
        return HERO_STYLE["mc_maxprob"]["color"]
    if crit.startswith("dropout_scratch"):
        return HERO_STYLE["dropout_scratch_sr"]["color"]
    return GREY


def fig_hero(ctxs: list[dict]) -> bytes | None:
    """SGR coverage vs r* per criterion, plus the no-guarantee empirical line."""
    if not any(c["opt"] for c in ctxs):
        return None
    fig, axes = panels(len(ctxs), height=3.6)
    for ax, ctx in zip(axes, ctxs, strict=False):
        rs = sorted(ctx["opt"])
        for crit in HERO_CRITERIA:
            ys = [opt_val(ctx, r, crit, "sgr cov_mean") for r in rs]
            if all(math.isnan(y) for y in ys):
                continue
            sd = [opt_val(ctx, r, crit, "sgr cov_std") for r in rs]
            st = HERO_STYLE[crit]
            ax.plot(rs, ys, color=st["color"], ls=st["ls"], lw=st["lw"], marker="o", ms=3.5, label=st["label"])
            ax.fill_between(
                rs,
                [max(y - s, 0.0) for y, s in zip(ys, sd, strict=False)],
                [min(y + s, 1.0) for y, s in zip(ys, sd, strict=False)],
                color=st["color"],
                alpha=0.12,
                lw=0,
            )
        emp = [opt_val(ctx, r, "ens_maxprob", "emp cov_mean") for r in rs]
        if not all(math.isnan(y) for y in emp):
            ax.plot(rs, emp, color=GREY, ls="--", label=f"{CRITERIA['ens_maxprob']['label']}, {RULE_LABELS['emp']}")
        ax.set_ylim(0, 1.02)
        style_ax(ax, "target risk r*", "coverage", ctx["name"] if len(ctxs) > 1 else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    return finish(fig, rect=(0, 0.1, 1, 1))


def hero_takeaway(ctxs: list[dict]) -> str:
    """Takeaway of the hero figure."""
    ctx = ctxs[0]
    rs = sorted(ctx["opt"])
    r = near(0.03, rs) or (rs[0] if rs else None)
    if r is None:
        return ""
    ens = opt_val(ctx, r, "ens_maxprob", "sgr cov_mean")
    base = opt_val(ctx, r, "sr_base", "sgr cov_mean")
    emp = opt_val(ctx, r, "ens_maxprob", "emp cov_mean")
    if math.isnan(ens) or math.isnan(base):
        return "SGR coverage grows with the target risk."
    s = f"At r* {r:g} SGR keeps {ens:.0%} coverage with the ensemble versus {base:.0%} for the base model"
    return s + (
        f", at a cost of {100 * (emp - ens):.0f} pts against the unguaranteed rule." if not math.isnan(emp) else "."
    )


def e3_reference(ctx: dict) -> str | None:
    """Most common reference name in the E3 table."""
    refs = [r["reference"] for r in ctx["e3"] if r.get("reference")]
    return max(set(refs), key=refs.count) if refs else None


def e3_rows(ctx: dict) -> list[dict[str, str]]:
    """E3 rows of the main reference only."""
    ref = e3_reference(ctx)
    return [r for r in ctx["e3"] if r.get("reference") == ref and r["criterion"] != ref]


def fig_gain(ctxs: list[dict]) -> bytes | None:
    """Paired coverage difference at the reference's risk, per criterion and r*."""
    ctxs = [c for c in ctxs if c["e3"]]
    if not ctxs:
        return None
    fig, axes = panels(len(ctxs))
    for ax, ctx in zip(axes, ctxs, strict=False):
        rows = e3_rows(ctx)
        rs = sorted({num(r["r_star"]) for r in rows})
        crits = sorted({r["criterion"] for r in rows}, key=lambda c: (c.startswith("ens_"), c))
        w = 0.7 / max(len(crits), 1)
        for i, crit in enumerate(crits):
            pts = [(rs.index(num(r["r_star"])), r) for r in rows if r["criterion"] == crit]
            is_ens = crit.startswith("ens_")
            col = crit_style(crit)["color"] if is_ens else "#c4c8cc"
            xs = [k + (i - len(crits) / 2) * w for k, _ in pts]
            ys = [100 * num(r["d_car_mean"]) for _, r in pts]
            es = [100 * num(r["d_car_std"]) for _, r in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt="o",
                ms=3.5 if is_ens else 2.5,
                color=col,
                lw=1,
                capsize=0,
                zorder=3 if is_ens else 2,
            )
            if crit == "ens_maxprob":
                for x, y, (_, r) in zip(xs, ys, pts, strict=False):
                    b = num(r["b_car"])
                    if not math.isnan(b):
                        ax.annotate(
                            f"{b:.0%}",
                            (x, y),
                            xytext=(0, 6),
                            textcoords="offset points",
                            ha="center",
                            fontsize=6.5,
                            color=col,
                        )
        ax.axhline(0, color="black", lw=0.8)
        ens_y = [100 * num(r["d_car_mean"]) for r in rows if r["criterion"].startswith("ens_")]
        if ens_y:
            ax.set_ylim(min(min(ens_y), 0) - 15, max(max(ens_y), 0) + 15)
        ax.set_xticks(range(len(rs)), [f"{r:g}" for r in rs])
        ref = e3_reference(ctx)
        style_ax(ax, "target risk r*", "coverage gain (pts)", f"{ctx['name'] + ', ' if len(ctxs) > 1 else ''}vs {crit_style(ref or '')['label']}")
    h = [
        plt.Line2D([], [], marker="o", ls="", color=crit_style("ens_maxprob")["color"]),
        plt.Line2D([], [], marker="o", ls="", color="#c4c8cc"),
    ]
    axes[0].legend(
        h,
        ["ensemble scores (label: share of splits won)", "other scores (axis clipped to ensemble range)"],
        frameon=False,
        fontsize=7,
        loc="lower right",
    )
    return finish(fig)


def gain_takeaway(ctxs: list[dict]) -> str:
    """Takeaway of the gain figure."""
    rows = [r for r in e3_rows(next(c for c in ctxs if c["e3"])) if r["criterion"] == "ens_maxprob"]
    if not rows:
        return ""
    r0 = min(rows, key=lambda r: num(r["r_star"]))
    return (
        f"At r* {num(r0['r_star']):g} the ensemble adds {100 * num(r0['d_car_mean']):.1f} pts of coverage "
        f"and wins {num(r0['b_car']):.0%} of paired splits."
    )


def cert_table(ctx: dict, r: float = 0.01) -> dict[str, float]:
    """Certified fraction per criterion at r* from E3."""
    return {
        r_["criterion"]: num(r_["own_cert"])
        for r_ in ctx["e3"]
        if abs(num(r_["r_star"]) - r) < 1e-9 and not math.isnan(num(r_["own_cert"]))
    }


def fig_fragile(ctxs: list[dict]) -> tuple[bytes, str] | None:
    """Certified fraction per criterion at r* 0.01 and a small table."""
    ctxs = [c for c in ctxs if c["e3"] and cert_table(c)]
    if not ctxs:
        return None
    fig, axes = panels(len(ctxs), height=3.6)
    tables = []
    for ax, ctx in zip(axes, ctxs, strict=False):
        cert = cert_table(ctx)
        top = sorted(cert, key=lambda c: -cert[c])[:12][::-1]
        cols = [family_color(c) for c in top]
        ax.barh(range(len(top)), [cert[c] for c in top], color=cols)
        ax.set_yticks(range(len(top)), [c for c in top], fontsize=7)
        ax.set_xlim(0, 1.05)
        style_ax(ax, "fraction of splits certified at r* 0.01", None, ctx["name"] if len(ctxs) > 1 else None)
        ax.grid(axis="y", alpha=0)
        tables.append(ctx)
    refresh_ticks(fig)
    return finish(fig), fragile_table(tables)


def fragile_table(ctxs: list[dict]) -> str:
    """HTML table criterion | cert @0.01 | cov @0.01 (per root)."""
    ref = e3_reference(ctxs[0])
    crits = [c for c in ["ens_maxprob", "mc_maxprob", "sr_base", "sr_dropout", ref, "dropout_scratch_maxprob"] if c][:6]
    head = "<tr><th>score</th>" + "".join(
        f"<th>{c['name']} cert</th><th>{c['name']} cov</th>"
        if len(ctxs) > 1
        else "<th>cert @0.01</th><th>cov @0.01</th>"
        for c in ctxs
    )
    body = ""
    for crit in dict.fromkeys(crits):
        cells = ""
        for ctx in ctxs:
            cert = cert_table(ctx).get(crit, NAN)
            cov = opt_val(ctx, 0.01, crit, "sgr cov_mean")
            cells += f"<td>{fmt_pct(cert)}</td><td>{fmt_pct(cov)}</td>"
        body += f"<tr><td>{crit_style(crit)["label"]}</td>{cells}</tr>"
    return f"<table>{head}</tr>{body}</table>"


def fmt_pct(x: float) -> str:
    """Format a fraction as a percentage, or a dash."""
    return "-" if math.isnan(x) else f"{x:.0%}"


def fragile_takeaway(ctxs: list[dict]) -> str:
    """Takeaway of the fragility figure."""
    ctx = next(c for c in ctxs if c["e3"] and cert_table(c))
    cert = cert_table(ctx)
    base = cert.get("sr_base", NAN)
    ens = cert.get("ens_maxprob", NAN)
    if math.isnan(base) or math.isnan(ens):
        low = min(cert, key=lambda c: cert[c])
        return f"At r* 0.01 even the best scores certify only part of the splits; {low} certifies {cert[low]:.0%}."
    return f"At r* 0.01 the base model certifies {base:.0%} of splits and the ensemble {ens:.0%}."


def mode_style(mode: str) -> str:
    """Color of a rule family: guaranteed green, conformal orange, others grey."""
    if mode in GUARANTEED:
        return MODE_COLORS[mode]
    return "#e8890c" if mode in CONFORMAL else GREY


def fig_rules(ctxs: list[dict]) -> bytes | None:
    """Coverage and violation rate per rule as horizontal bars, at r* 0.01 and 0.03."""
    ctxs = [c for c in ctxs if c["opt"]]
    rs = [r for r in (0.01, 0.03) if any(near(r, list(c["opt"])) for c in ctxs)]
    if not ctxs or not rs:
        return None
    order = ["sgr", "LTT Bonf", "LTT FS", "FS split", "Conf LAC", "Conf APS", "Chow raw", "Chow TS", "emp"]
    pal = {"g": ("#0b5d46", "#8fd1b4"), "c": ("#e65100", "#ffcc99"), "n": ("#555555", "#c8c8c8")}
    fig, axs = plt.subplots(len(ctxs), len(rs), figsize=(10, 3.8 * len(ctxs)), squeeze=False)
    for i, ctx in enumerate(ctxs):
        for j, r in enumerate(rs):
            ax = axs[i][j]
            rr = near(r, list(ctx["opt"]))
            for k, mode in enumerate(order):
                cls = "g" if mode in GUARANTEED else "c" if mode in CONFORMAL else "n"
                for crit, dy, shade in (("ens_maxprob", -0.19, 0), ("sr_base", 0.19, 1)):
                    c = opt_val(ctx, rr, crit, f"{mode} cov_mean") if rr else NAN
                    v = opt_val(ctx, rr, crit, f"{mode} viol_mean") if rr else NAN
                    if math.isnan(c):
                        continue
                    ax.barh(k + dy, c, 0.34, color=pal[cls][shade])
                    txt = f"{c:.2f}" + (f"  viol {v:.0%}" if not math.isnan(v) and v > 0.01 else "")
                    ax.text(c + 0.01, k + dy, txt, va="center", fontsize=7.5)
            ax.set_yticks(range(len(order)), [RULE_LABELS[m] for m in order], fontsize=8)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_xlim(0, 1.3)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.axhline(3.5, color="#999", lw=0.6)
            ax.axhline(5.5, color="#999", lw=0.6)
            title = f"r* {r:g}" + (f", {ctx['name']}" if len(ctxs) > 1 else "")
            style_ax(ax, "coverage", None, title)
            ax.grid(axis="y", alpha=0)
    h = [plt.Rectangle((0, 0), 1, 1, color="#555555"), plt.Rectangle((0, 0), 1, 1, color="#c8c8c8")]
    fig.legend(h, [f'{CRITERIA["ens_maxprob"]["label"]} (dark)', f'{CRITERIA["sr_base"]["label"]} (light)'], loc="lower center", ncol=2, frameon=False, fontsize=8)
    refresh_ticks(fig)
    return finish(fig, rect=(0, 0.05 if len(ctxs) == 1 else 0.03, 1, 1))


def rules_takeaway(ctxs: list[dict]) -> str:
    """Takeaway of the rules figure: best guaranteed mode for ens_maxprob at r* 0.03."""
    ctx = ctxs[0]
    r = near(0.03, list(ctx["opt"])) or next(iter(ctx["opt"]), None)
    if r is None:
        return ""
    cov = {m: opt_val(ctx, r, "ens_maxprob", f"{m} cov_mean") for m in GUARANTEED}
    cov = {m: v for m, v in cov.items() if not math.isnan(v)}
    if not cov:
        return ""
    best = max(cov, key=lambda m: cov[m])
    if best == "sgr" or "sgr" not in cov:
        return f"At r* {r:g} SGR is already the guaranteed rule with the highest coverage ({cov[best]:.0%})."
    return (
        f"At r* {r:g} {best} has the highest guaranteed coverage, {100 * (cov[best] - cov['sgr']):.1f} pts above SGR."
    )


def fig_scores(ctx: dict) -> bytes | None:
    """AURC and AUGRC per score, and coverage difference to the reference (E4)."""
    rows = ctx["e4"]
    if not rows:
        return None
    r0 = min(num(r["r_star"]) for r in rows)
    sel = [r for r in rows if num(r["r_star"]) == r0]
    sel = sorted(sel, key=lambda r: num(r["aurc_x1000_mean"]))[:12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = list(range(len(sel)))
    has_gr = any(not math.isnan(num(r.get("augrc_x1000_mean"))) for r in sel)
    h = 0.38 if has_gr else 0.7
    aurc = [num(r["aurc_x1000_mean"]) for r in sel]
    augrc = [num(r.get("augrc_x1000_mean")) for r in sel]
    a1.barh([k - (h / 2 if has_gr else 0) for k in y], aurc, h, color="#1e88e5", label="AURC x1000")
    if has_gr:
        a1.barh([k + h / 2 for k in y], augrc, h, color="#fb8c00", label="AUGRC x1000")
    vals = sorted(v for v in aurc if not math.isnan(v))
    med = vals[len(vals) // 2] if vals else 1.0
    if vals and vals[-1] > 4 * med:
        lim = 2.5 * med
        a1.set_xlim(0, lim)
        for k, v in enumerate(aurc):
            if v > lim:
                a1.text(
                    lim * 0.98,
                    k - (h / 2 if has_gr else 0),
                    f"{v:.1f} >",
                    ha="right",
                    va="center",
                    fontsize=7,
                    color="white",
                )
    a1.set_yticks(y, [crit_style(r["score"])["label"] for r in sel], fontsize=7)
    a1.set_ylim(len(sel) - 0.5, -0.5)
    a1.legend(frameon=False, fontsize=7, loc="upper right")
    style_ax(a1, "area under the curve, x1000 (lower is better)", None, "Ranking quality")
    a1.grid(axis="y", alpha=0)
    cols = ["#2e9e6f" if truthy(r.get("beats_sr", "0")) else GREY for r in sel]
    d = [100 * num(r["d_car_mean"]) for r in sel]
    a2.barh(y, d, color=cols, alpha=0.5, height=0.6)
    a2.errorbar(d, y, xerr=[100 * num(r["d_car_std"]) for r in sel], fmt="x", color="#333", ms=5, lw=0.8, capsize=0)
    a2.axvline(0, color="#777", lw=0.8)
    style_ax(a2, "coverage gain over max-prob (pts)", None, f"Coverage at the same risk, r* {r0:g}")
    a2.grid(axis="y", alpha=0)
    refresh_ticks(fig)
    return finish(fig)


def scores_takeaway(ctxs: list[dict]) -> str:
    """Takeaway of the scores figure."""
    parts = []
    for ctx in ctxs:
        if not ctx["e4"] or "beats_sr" not in ctx["e4"][0]:
            continue
        r0 = min(num(r["r_star"]) for r in ctx["e4"])
        sel = [r for r in ctx["e4"] if num(r["r_star"]) == r0]
        k = sum(truthy(r["beats_sr"]) for r in sel)
        parts.append(f"{k} of {len(sel)}" + (f" ({ctx['name']})" if len(ctxs) > 1 else ""))
    return (
        f"Scores that beat max-prob in the paired comparison at the smallest r*: {', '.join(parts)}." if parts else ""
    )


def shift_series(ctx: dict, corr: str, crit: str, col: str) -> tuple[list[float], list[float]]:
    """Severity and column values of one criterion in one shift file."""
    rows = sorted((r for r in ctx["shift"][corr] if r["criterion"] == crit), key=lambda r: num(r["severity"]))
    return [num(r["severity"]) for r in rows], [num(r.get(col)) for r in rows]


def fig_shift(ctxs: list[dict], rstar: float = 0.03) -> bytes | None:
    """SGR test risk vs corruption severity."""
    ctxs = [c for c in ctxs if c["shift"]]
    if not ctxs:
        return None
    col = f"r*={rstar:g} sgr risk_mean"
    corrs = [c for c in CORRUPTIONS if any(c in ctx["shift"] for ctx in ctxs)]
    fig, axs = plt.subplots(len(ctxs), len(corrs), figsize=(10, 2.7 * len(ctxs)), squeeze=False)
    for i, ctx in enumerate(ctxs):
        for j, corr in enumerate(corrs):
            ax = axs[i][j]
            if corr in ctx["shift"]:
                for crit in SHIFT_CRITERIA:
                    xs, ys = shift_series(ctx, corr, crit, col)
                    if xs and not all(math.isnan(y) for y in ys):
                        st = HERO_STYLE[crit]
                        ax.plot(
                            xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"], marker="o", ms=3, label=st["label"]
                        )
            ax.axhline(rstar, color="#c0392b", ls=":", lw=1, label=f"target r* {rstar:g}")
            title = corr.replace("_", " ") + (f" ({ctx['name']})" if len(ctxs) > 1 else "")
            style_ax(ax, "severity", "test risk" if j == 0 else None, title)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    return finish(fig, rect=(0, 0.07, 1, 1))


def shift_takeaway(ctxs: list[dict], rstar: float = 0.03) -> str:
    """Name the corruption where the ensemble first exceeds r*, with its lowest failing severity."""
    ctx = next(c for c in ctxs if c["shift"])
    col = f"r*={rstar:g} sgr risk_mean"
    crit = next(
        (
            c
            for c in SHIFT_CRITERIA[1:] + SHIFT_CRITERIA[:1]
            if any(r["criterion"] == c for rows in ctx["shift"].values() for r in rows)
        ),
        None,
    )
    if crit is None:
        return ""
    worst: tuple[float, str] | None = None
    for corr in ctx["shift"]:
        xs, ys = shift_series(ctx, corr, crit, col)
        fails = [x for x, y in zip(xs, ys, strict=False) if not math.isnan(y) and y > rstar]
        if fails and (worst is None or min(fails) < worst[0]):
            worst = (min(fails), corr.replace("_", " "))
    if worst is None:
        return f"{crit_style(crit)["label"]} keeps the SGR test risk below r* {rstar:g} at every severity on all corruptions."
    return (
        f"SGR thresholds tuned in-distribution break under shift: at r* {rstar:g}, {crit_style(crit)["label"]} exceeds the target "
        f"already at severity {worst[0]:g} ({worst[1]})."
    )


def fig_cost(rows: list[dict[str, str]]) -> tuple[bytes, str] | None:
    """Coverage vs ensemble members or MC samples at the smallest r*."""
    rows = [r for r in rows if r.get("family") and not math.isnan(num(r.get("cov_mean")))]
    if not rows:
        return None
    r0 = min(num(r["r_star"]) for r in rows)
    rows = [r for r in rows if num(r["r_star"]) == r0]
    fig, (ax,) = panels(1, height=3.4)
    fam_col = {
        f: c
        for f, c in zip(sorted({r["family"] for r in rows}), ["#1e88e5", "#16a085", "#9b59b6", "#fb8c00"], strict=False)
    }
    ls = {"maxprob": "-", "epistemic": "--"}
    best = (NAN, "")
    for fam in fam_col:
        for crit in sorted({r["criterion"] for r in rows if r["family"] == fam}):
            pts = sorted(
                (num(r["size"]), num(r["cov_mean"])) for r in rows if r["family"] == fam and r["criterion"] == crit
            )
            ax.plot(*zip(*pts), color=fam_col[fam], ls=ls.get(crit, ":"), marker="o", ms=3.5, label=f"{fam}, {crit}")
            if math.isnan(best[0]) or pts[-1][1] > best[0]:
                best = (pts[-1][1], f"{fam}, {crit}, size {pts[-1][0]:g}")
    ax.set_xscale("log", base=2)
    style_ax(ax, "ensemble members / MC samples (log2)", f"coverage at r* {r0:g}")
    ax.legend(frameon=False, fontsize=8)
    refresh_ticks(fig)
    return finish(
        fig
    ), f"More members or samples raise coverage at r* {r0:g}; the best point is {best[1]} with {best[0]:.0%}."


# ------------------------------------------------------------------------------------------------------------- HTML
CSS = """
:root { --bg:#fafaf8; --fg:#1d2125; --mut:#5f6b76; --line:#e3e5e8; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 'Fira Sans',system-ui,-apple-system,'Segoe UI',sans-serif; }
main { max-width:1000px; margin:0 auto; padding:28px 16px 48px; }
h1 { font-size:28px; font-weight:600; margin:0 0 4px; }
h2 { font-size:19px; font-weight:600; margin:34px 0 4px; }
.sub { color:var(--mut); margin:0 0 18px; }
.take { margin:0 0 10px; }
.kpis { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 0; }
.kpi { flex:1 1 200px; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.kpi b { display:block; font-size:26px; font-weight:600; line-height:1.2; }
.kpi span { color:var(--mut); font-size:13px; }
img { width:100%; height:auto; display:block; background:#fff; border:1px solid var(--line); border-radius:6px; }
.cap { color:var(--mut); font-size:13px; margin:8px 0 4px; }
table { border-collapse:collapse; margin:10px 0 0; font-size:13px; }
th,td { padding:3px 12px; text-align:right; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
ul { margin:6px 0 0; padding-left:20px; }
"""


def img_tag(png: bytes, alt: str) -> str:
    """Embed PNG bytes as a base64 image."""
    return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def section(title: str, take: str, body: str) -> str:
    """One section: heading, one-sentence takeaway, body (figure and table)."""
    return f'<section><h2>{title}</h2><p class="take">{take}</p>{body}</section>'


def kpi_cards(ctxs: list[dict]) -> str:
    """KPI cards, with per-root values joined by a slash."""

    def join(vals: list[str]) -> str:
        return " / ".join(vals)

    cards = []
    gains = []
    for c in ctxs:
        e, b = opt_val(c, 0.01, "ens_maxprob", "sgr cov_mean"), opt_val(c, 0.01, "sr_base", "sgr cov_mean")
        gains.append("n/a" if math.isnan(e - b) else f"+{100 * (e - b):.0f} pts")
    if any(g != "n/a" for g in gains):
        cards.append((join(gains), "ensemble coverage gain at r* 0.01"))
    certs = [cert_table(c).get("sr_base", NAN) for c in ctxs]
    if any(not math.isnan(x) for x in certs):
        cards.append((join([fmt_pct(x) for x in certs]), "splits where the base model certifies at r* 0.01"))
    best = []
    for c in ctxs:
        r = near(0.03, list(c["opt"]))
        cov = {m: opt_val(c, r, "ens_maxprob", f"{m} cov_mean") for m in GUARANTEED} if r else {}
        cov = {m: v for m, v in cov.items() if not math.isnan(v)}
        best.append(f"{max(cov, key=lambda m: cov[m])} {max(cov.values()):.0%}" if cov else "n/a")
    if any(b != "n/a" for b in best):
        cards.append((join(best), "best guaranteed rule, ensemble at r* 0.03"))
    beats = []
    for c in ctxs:
        if c["e4"] and "beats_sr" in c["e4"][0]:
            r0 = min(num(r["r_star"]) for r in c["e4"])
            beats.append(str(sum(truthy(r["beats_sr"]) for r in c["e4"] if num(r["r_star"]) == r0)))
    if beats:
        cards.append((join(beats), "scores that beat max-prob (E4)"))
    return (
        '<div class="kpis">'
        + "".join(f'<div class="kpi"><b>{v}</b><span>{lab}</span></div>' for v, lab in cards)
        + "</div>"
    )


def build_html(ctxs: list[dict], cost: list[dict[str, str]]) -> str:
    """Assemble all sections."""
    names = ", ".join(c["name"] for c in ctxs)
    secs: list[str] = []

    def add(title: str, png: bytes | None, take_fn, extra: str = "", note: str = "") -> None:
        if png is None:
            print(f"Skipping '{title}': {note or 'input CSVs missing'}.")
            return
        secs.append(section(title, take_fn(), img_tag(png, title) + extra))

    add("Coverage vs r*", fig_hero(ctxs), lambda: hero_takeaway(ctxs))
    add("Ensemble gain is consistent", fig_gain(ctxs), lambda: gain_takeaway(ctxs), note="matched/e3.csv missing")
    frag = fig_fragile(ctxs)
    add(
        "Fragile at r* = 0.01",
        frag[0] if frag else None,
        lambda: fragile_takeaway(ctxs),
        frag[1] if frag else "",
        "matched/e3.csv missing",
    )
    add("Which rule?", fig_rules(ctxs), lambda: rules_takeaway(ctxs))
    e4 = [(c, fig_scores(c)) for c in ctxs]
    e4 = [(c, p) for c, p in e4 if p]
    if e4:
        body = "".join(
            (f'<p class="cap">{c["name"]}</p>' if len(ctxs) > 1 else "") + img_tag(p, "scores") for c, p in e4
        )
        secs.append(section("No score beats max-prob", scores_takeaway([c for c, _ in e4]), body))
    else:
        print("Skipping 'No score beats max-prob': matched/e4.csv missing.")
    add("Under shift", fig_shift(ctxs), lambda: shift_takeaway(ctxs), note="shift_*.csv missing")
    if cost:
        res = fig_cost(cost)
        if res:
            secs.append(section("Cost", res[1], img_tag(res[0], "cost")))
    steps = (
        "<ul><li>Label-budget sweep (n_cal 500 to 5000).</li><li>ResNet cost sweep.</li>"
        "<li>Tighter bounds and better scores.</li></ul>"
    )
    secs.append(f"<section><h2>Next steps</h2>{steps}</section>")
    sub = f"CIFAR-10; 5 seeds x 10 splits of 5k/5k; delta 0.001; {names}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Selective prediction summary</title>"
        '<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;600&display=swap" rel="stylesheet">'
        f"<style>{CSS}</style></head><body><main>"
        f'<h1>Selective prediction with guaranteed risk: summary</h1><p class="sub">{sub}</p>'
        f"{kpi_cards(ctxs)}{''.join(secs)}</main></body></html>"
    )


# ------------------------------------------------------------------------------------------------------------- PDF
def find_chrome(explicit: str | None) -> str | None:
    """Locate a Chrome/Chromium executable (explicit path, macOS/Windows default paths, then PATH)."""
    candidates = [explicit] if explicit else []
    candidates += [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def render_pdf(html_path: Path, pdf_path: Path, chrome: str | None) -> bool:
    """Print the summary to PDF with headless Chrome; returns False if Chrome is missing or fails."""
    exe = find_chrome(chrome)
    if exe is None:
        print("Chrome not found, skipping the PDF (pass --chrome PATH).")
        return False
    cmd = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)  # noqa: S603
    except (OSError, subprocess.SubprocessError) as e:
        print(f"PDF generation failed: {e}")
        return False
    return pdf_path.exists()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", default=[Path("results")], help="One or two result roots.")
    p.add_argument("--names", nargs="+", default=None, help="Labels of the roots (default: directory names).")
    p.add_argument("--cost", type=Path, default=None, help="Optional cost table.csv.")
    p.add_argument("--out", type=Path, default=Path("results/summary.html"))
    p.add_argument("--pdf", action="store_true", help="Also print a PDF next to the HTML with headless Chrome.")
    p.add_argument("--chrome", default=None, help="Path of the Chrome/Chromium executable.")
    return p.parse_args()


def main() -> None:
    """Build the summary."""
    global HAS_FIRA  # noqa: PLW0603
    args = parse_args()
    HAS_FIRA = setup_fonts()
    roots = args.results[:2]
    names = args.names or [r.resolve().name for r in roots]
    ctxs = [load_root(r, n) for r, n in zip(roots, names, strict=False)]
    cost = read_csv(args.cost) if args.cost else []
    html = build_html(ctxs, cost)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {args.out}")
    if args.pdf:
        pdf = args.out.with_suffix(".pdf")
        if render_pdf(args.out, pdf, args.chrome):
            print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
