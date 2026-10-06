"""Self-contained HTML (and optionally PDF) results report of the SGR selective prediction experiment.

Inputs, all under ``--results`` (written by the evaluation scripts; only the standard library and numpy are used here):

- ``table.csv``, ``table_sgr.csv``, ``table_coverage.csv``, ``fixed_threshold.csv``: reproduction tables (cells are
  strings such as ``0.8078 +- 0.0130``).
- ``shift/``: ``id_aurc.csv``, ``id_thresholds.csv``, ``ood_*.csv``, ``mixed_*.csv``, ``sgr_check.csv``, ``shift_*.csv``.
- ``options/``: ``id_r*.csv``, ``ood_*.csv``, ``shift_*_r*.csv`` (all accept/abstain modes).
- ``calibration/``: ``clean_calibration.csv``, ``ranking_*.csv``, ``shift_*.csv``.
- every ``*.png`` under ``results``, ``results/shift``, ``results/options``, ``results/calibration``, ``results/summary``.
- ``report_text.py`` next to this script with ``TITLE``, ``SUBTITLE`` and ``SECTIONS`` (key -> HTML fragment). Missing
  text becomes a ``TODO`` placeholder paragraph.

Outputs:

- ``report.html`` (default ``<results>/report.html``): one offline-capable page (figures as base64, all csv data as a JSON
  blob, tables rendered client-side with vanilla JS; only the Fira Sans stylesheet is loaded from Google Fonts).
- ``report.pdf`` next to it with ``--pdf``, printed by headless Chrome (all tabs of the tabbed views are printed).

The table of main scores (lowest clean AURC criterion per method) is also printed to stdout.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from evaluate_shift import CRITERIA, _METHOD_STYLE  # noqa: PLC2701
from plot_results import CORRUPTIONS, MODES, SPECIAL_METHODS, method_of

from sgr_experiment.utils import EXPERIMENT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import report_text  # ty: ignore[unresolved-import]
except ImportError:  # the text module is optional while it is being written
    report_text = None

SECTION_KEYS = ["overview", "setup", "reproduction", "clean", "rules", "cliff", "ood", "shift", "calibration", "verdict", "next"]
SECTION_TITLES = {
    "overview": "Overview",
    "setup": "Setup",
    "reproduction": "Reproduction of Table 1",
    "clean": "Clean-data ranking",
    "rules": "Threshold rules",
    "cliff": "The certification cliff",
    "ood": "Out-of-distribution detection",
    "shift": "Covariate shift",
    "calibration": "Calibration",
    "verdict": "Verdict",
    "next": "Next steps",
}
SECTION_VIEWS = {
    "reproduction": ["repro"],
    "clean": ["clean", "main"],
    "rules": ["rules"],
    "cliff": ["cert"],
    "ood": ["ood"],
    "shift": ["shift", "shiftopt"],
    "calibration": ["calib"],
}
FIGURE_PLACEMENT = {
    "reproduction": ["risk_coverage.png"],
    "clean": ["summary/main_scores.png", "shift/id_risk_coverage*.png"],
    "rules": ["summary/modes.png"],
    "ood": ["shift/ood_acceptance*.png"],
    "shift": ["summary/shift_risk.png", "shift/shift_risk*.png"],
    "calibration": ["summary/calibration.png", "calibration/ece_vs_severity.png"],
}
SKIP_FIGURES = {"options/coverage_by_mode.png"}
NO_HEAT_FRAMES = {"table", "table_sgr", "table_coverage", "fixed_threshold"}
GUARANTEED_MODES = ["sgr", "LTT Bonf", "LTT FS"]
VIOL_FLAG = 0.01
KEY_COLUMNS = {"criterion", "source", "severity", "r*", "r_star", "probabilities", "table"}

_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PAIR = re.compile(rf"^({_NUM}) \+- ({_NUM})$")
_PCT = re.compile(rf"^({_NUM})%$")
_NUMBER = re.compile(rf"^{_NUM}$")

Frame = dict
Frames = dict[str, Frame]


def _round(x: float) -> float:
    """Round to six significant digits (keeps the embedded JSON small); NaN becomes NaN."""
    return float(f"{x:.6g}")


def fmt_for(rel: str, name: str) -> dict | None:
    """Display format of a logical column: decimals ``d``, percent flag, direction (-1 lower, 1 higher, 0 neutral).

    Args:
        rel: Path of the frame relative to the results directory, without suffix.
        name: Logical column name (``_mean``/``_std`` already merged).

    Returns:
        A dict ``{"d", "pct", "dir", "viol"}``, or None for key columns that are shown verbatim.
    """
    n = name.lower()
    if n in KEY_COLUMNS or n == "c_or_r_star":
        return None
    tokens = re.split(r"[ _]", n)
    last = tokens[-1]
    f: dict
    if "viol" in n or "uncertified" in n:
        f = {"d": 0, "pct": True, "dir": -1, "viol": True}
    elif rel.startswith("options/ood_") or "accepted" in n:
        f = {"d": 3, "pct": False, "dir": -1}
    elif rel.startswith("calibration/ranking") or "aurc" in n:
        f = {"d": 2, "pct": False, "dir": -1}
    elif "auroc" in n or "accuracy" in n:
        f = {"d": 3, "pct": False, "dir": 1}
    elif "cov_at" in n or last in ("cov", "coverage"):
        f = {"d": 3, "pct": False, "dir": 1}
    elif last == "risk":
        f = {"d": 4, "pct": False, "dir": -1}
    elif {"ece", "nll", "brier"} & set(tokens):
        f = {"d": 4, "pct": False, "dir": -1}
    elif last == "bound":
        f = {"d": 4, "pct": False, "dir": 0}
    elif "differing" in n or n == "t":
        f = {"d": 3, "pct": False, "dir": 0}
    elif rel == "fixed_threshold":
        f = {"d": 4, "pct": False, "dir": 0}
    else:
        return None
    if n.startswith("paper"):
        f["dir"] = 0
    f.setdefault("viol", False)
    return f


def _parse_cell(v: str) -> object:
    """Parse one csv cell into None, a float, ``[mean, std]``, ``("pct", fraction)`` or the raw string."""
    if v == "":
        return None
    m = _PAIR.match(v)
    if m:
        return [_round(float(m.group(1))), _round(float(m.group(2)))]
    m = _PCT.match(v)
    if m:
        return ("pct", _round(float(m.group(1)) / 100.0))
    if _NUMBER.match(v):
        return _round(float(v))
    return v


def load_frame(path: Path, rel: str) -> Frame:
    """Read a csv file into a frame with merged ``_mean``/``_std`` columns, formats and the raw csv text."""
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    raw_rows = [r for r in reader if r]
    hset = set(header)
    plan: list[tuple[str, int, int | None]] = []
    for j, h in enumerate(header):
        if h.endswith("_std") and h[:-4] + "_mean" in hset:
            continue
        if h.endswith("_mean") and h[:-5] + "_std" in hset:
            plan.append((h[:-5], j, header.index(h[:-5] + "_std")))
        else:
            plan.append((h, j, None))
    cols = [p[0] for p in plan]
    parsed: list[list[object]] = []
    for r in raw_rows:
        row: list[object] = []
        for _, j, k in plan:
            if k is None:
                row.append(_parse_cell(r[j]))
            else:
                if r[j] == "":
                    row.append(None)
                else:
                    std = float(r[k]) if r[k] != "" else 0.0
                    row.append([_round(float(r[j])), _round(std)])
        parsed.append(row)
    fmts: list[dict | None] = []
    for ci, name in enumerate(cols):
        values = [row[ci] for row in parsed]
        if any(isinstance(v, str) for v in values):  # text column: keep the original strings
            for ri, row in enumerate(parsed):
                orig = raw_rows[ri][plan[ci][1]]
                row[ci] = orig if orig != "" else None
            fmts.append(None)
            continue
        pct = any(isinstance(v, tuple) for v in values)
        for row in parsed:
            if isinstance(row[ci], tuple):
                row[ci] = row[ci][1]
        f = fmt_for(rel, name)
        if pct and f is None:
            f = {"d": 0, "pct": True, "dir": -1, "viol": True}
        fmts.append(f)
    return {"cols": cols, "fmts": fmts, "rows": parsed, "csv": text, "heat": rel not in NO_HEAT_FRAMES}


def read_frames(results: Path) -> Frames:
    """Load every csv below ``results`` (but not below ``summary``) keyed by its path without suffix."""
    frames: Frames = {}
    for p in sorted(results.rglob("*.csv")):
        rel = p.relative_to(results).with_suffix("").as_posix()
        if rel.startswith("summary/"):
            continue
        frames[rel] = load_frame(p, rel)
    return frames


def cell_mean(cell: object) -> float | None:
    """Mean of a cell (number or ``[mean, std]``), None if missing or text."""
    if isinstance(cell, list):
        return float(cell[0])
    if isinstance(cell, (int, float)):
        return float(cell)
    return None


def _same(a: object, b: object) -> bool:
    """Equality that treats 1, 1.0 and "1" alike."""
    try:
        return float(a) == float(b)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return str(a) == str(b)


def lookup(frames: Frames, rel: str, col: str, **where: object) -> float | None:
    """Mean of column ``col`` in the first row of frame ``rel`` whose key columns equal ``where`` (as strings)."""
    fr = frames.get(rel)
    if fr is None or col not in fr["cols"]:
        return None
    idx = {c: i for i, c in enumerate(fr["cols"])}
    for row in fr["rows"]:
        if all(_same(row[idx[k.replace("_star", "*")]], v) for k, v in where.items()):
            return cell_mean(row[idx[col]])
    return None


def compute_main_scores(frames: Frames, rs: list[str]) -> Frame:
    """Main score per method (criterion with the lowest clean AURC) and the summary columns as a derived frame."""
    aurc = frames["shift/id_aurc"]
    ci = aurc["cols"].index("criterion")
    ai = aurc["cols"].index("AURC x1000")
    best: dict[str, str] = {}
    for row in sorted(aurc["rows"], key=lambda r: cell_mean(r[ai]) or math.inf):
        best.setdefault(method_of(row[ci]), row[ci])
    low = {"d": 4, "pct": False, "dir": -1, "viol": False}
    cov = {"d": 3, "pct": False, "dir": 1, "viol": False}
    spec: list[tuple[str, dict | None]] = [
        ("method", None),
        ("criterion", None),
        ("AURC x1000", {"d": 2, "pct": False, "dir": -1, "viol": False}),
        ("E-AURC x1000", {"d": 2, "pct": False, "dir": -1, "viol": False}),
        ("accuracy", cov),
        *[(f"SGR cov r*={r}", cov) for r in rs[:3]],
        ("uncertified r*=0.01", {"d": 0, "pct": True, "dir": -1, "viol": False}),
        ("SVHN AUROC", cov),
        ("CIFAR-100 AUROC", cov),
        ("ECE raw", low),
        ("ECE TS", low),
        *[(f"SGR risk r*=0.03 sev {s}", low) for s in (1, 3, 5)],
    ]
    rows = []
    for m, c in best.items():
        row: list[object] = [
            m,
            c,
            lookup(frames, "shift/id_aurc", "AURC x1000", criterion=c),
            lookup(frames, "shift/id_aurc", "E-AURC x1000", criterion=c),
            lookup(frames, "shift/id_aurc", "accuracy", criterion=c),
        ]
        row += [lookup(frames, f"options/id_r{r}", "sgr cov", criterion=c) for r in rs[:3]]
        row.append(lookup(frames, "shift/sgr_check", "uncertified r*=0.01", criterion=c))
        row.append(lookup(frames, "shift/ood_svhn", "AUROC", criterion=c))
        row.append(lookup(frames, "shift/ood_cifar100", "AUROC", criterion=c))
        row.append(lookup(frames, "calibration/clean_calibration", "ECE raw", source=m))
        row.append(lookup(frames, "calibration/clean_calibration", "ECE TS", source=m))
        for s in (1, 3, 5):
            vals = [lookup(frames, f"shift/shift_{k}", "r*=0.03 sgr risk", severity=s, criterion=c) for k in CORRUPTIONS]
            vals = [v for v in vals if v is not None]
            row.append(sum(vals) / len(vals) if vals else None)
        rows.append([_round(v) if isinstance(v, float) else v for v in row])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([n for n, _ in spec])
    w.writerows(["" if v is None else v for v in row] for row in rows)
    return {"cols": [n for n, _ in spec], "fmts": [f for _, f in spec], "rows": rows, "csv": buf.getvalue(), "heat": True}


def print_main_scores(fr: Frame) -> None:
    """Print the main-score table to stdout."""
    print("Main scores (best criterion per method by clean AURC):")
    print("  " + " | ".join(fr["cols"]))
    for row in fr["rows"]:
        print("  " + " | ".join("n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row))


def headline_cards(frames: Frames, main: Frame, rs: list[str]) -> list[tuple[str, str, str]]:
    """The five overview cards as ``(label, value, detail)`` computed from the data."""
    aurc = frames["shift/id_aurc"]
    ci, ai = aurc["cols"].index("criterion"), aurc["cols"].index("AURC x1000")
    top = min(aurc["rows"], key=lambda r: cell_mean(r[ai]) or math.inf)
    cards = [("Best clean AURC", f"{cell_mean(top[ai]):.2f}", f"{top[ci]} (x 1000)")]
    opt = frames[f"options/id_r{rs[0]}"]
    oc, sc = opt["cols"].index("criterion"), opt["cols"].index("sgr cov")
    topc = max(opt["rows"], key=lambda r: cell_mean(r[sc]) if cell_mean(r[sc]) is not None else -1)
    cards.append((f"Best SGR coverage, r* = {rs[0]}", f"{cell_mean(topc[sc]):.3f}", str(topc[oc])))
    sv = frames["shift/ood_svhn"]
    vc, va = sv["cols"].index("criterion"), sv["cols"].index("AUROC")
    tops = max(sv["rows"], key=lambda r: cell_mean(r[va]) or -1)
    cards.append(("Best SVHN AUROC", f"{cell_mean(tops[va]):.3f}", str(tops[vc])))
    bad = total = 0
    for r in rs:
        fr = frames[f"options/id_r{r}"]
        for m in GUARANTEED_MODES:
            j = fr["cols"].index(f"{m} viol")
            for row in fr["rows"]:
                v = cell_mean(row[j])
                if v is not None:
                    total += 1
                    bad += v > VIOL_FLAG
    cards.append(("Guaranteed rules violating", f"{100 * bad / max(total, 1):.0f}%", f"{bad} of {total} cells (sgr, LTT Bonf, LTT FS; viol > 1%)"))
    k = main["cols"].index("SGR risk r*=0.03 sev 5")
    vals = [row[k] for row in main["rows"] if row[k] is not None]
    cards.append(("SGR risk under severity 5", f"{sum(vals) / max(len(vals), 1):.4f}", "mean over main scores and corruptions, r* = 0.03"))
    return cards


def method_colors() -> dict[str, str]:
    """Color per method key (post-training methods from ``_METHOD_STYLE``, the rest from ``CRITERIA``)."""
    colors = {m: v[1] for m, v in _METHOD_STYLE.items()}
    colors["sr_base"] = CRITERIA["sr_base"]["color"]
    colors["sr_dropout"] = CRITERIA["sr_dropout"]["color"]
    colors["mc"] = CRITERIA["mc_maxprob"]["color"]
    colors["ens"] = CRITERIA["ens_total"]["color"]
    colors["swa"] = CRITERIA["swa_maxprob"]["color"]
    return colors


def collect_figures(results: Path) -> dict[str, list[Path]]:
    """Assign every png to a section (or ``appendix``); ``SKIP_FIGURES`` are dropped."""
    allpng = {p.relative_to(results).as_posix(): p for p in sorted(results.rglob("*.png")) if p.parent.relative_to(results).as_posix() in {".", "shift", "options", "calibration", "summary"}}
    placed: set[str] = set()
    out: dict[str, list[Path]] = {}
    for sec, patterns in FIGURE_PLACEMENT.items():
        for pat in patterns:
            for rel in sorted(k for k in allpng if Path(k).match(pat) and (("/" in pat) == ("/" in k)) and k.startswith(pat.split("*", maxsplit=1)[0])):
                if rel not in placed:
                    placed.add(rel)
                    out.setdefault(sec, []).append(allpng[rel])
    out["appendix"] = [p for k, p in allpng.items() if k not in placed and k not in SKIP_FIGURES]
    return out


def figure_html(path: Path) -> str:
    """A figure element with the png embedded as a base64 data URI."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    cap = html.escape(path.stem.replace("_", " "))
    return f'<figure class="fig"><img src="data:image/png;base64,{data}" alt="{cap}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


def get_text(key: str) -> str:
    """Narrative fragment of a section, or a TODO placeholder paragraph."""
    sections = getattr(report_text, "SECTIONS", {}) if report_text is not None else {}
    return sections.get(key) or f'<p class="todo">TODO: text for {key}</p>'


def git_commit() -> str:
    """Short git commit hash of the checkout, empty if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=EXPERIMENT_DIR, capture_output=True, text=True, check=True, timeout=10)  # noqa: S607
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def build_meta(frames: Frames) -> dict:
    """Metadata the client-side views need (criteria order, risk levels, option lists)."""
    rs = sorted({p.split("id_r")[1] for p in frames if p.startswith("options/id_r")}, key=float)
    ood_cols = frames["options/ood_svhn"]["cols"]
    ood_rs = sorted({m.group(1) for c in ood_cols if (m := re.search(r"r\*=([\d.]+)$", c))}, key=float)
    mixed = frames["shift/mixed_svhn"]
    mixed_rs = sorted({str(r[mixed["cols"].index("r*")]) for r in mixed["rows"]}, key=float)
    shift_rs = sorted({m.group(1) for c in frames["shift/shift_contrast"]["cols"] if (m := re.match(r"r\*=([\d.]+) ", c))}, key=float)
    aurc = frames["shift/id_aurc"]
    crit = [r[aurc["cols"].index("criterion")] for r in aurc["rows"]]
    cal = frames["calibration/clean_calibration"]
    src = [r[cal["cols"].index("source")] for r in cal["rows"]]
    sevs = sorted({str(r[frames["shift/shift_contrast"]["cols"].index("severity")]) for r in frames["shift/shift_contrast"]["rows"]}, key=float)
    return {
        "modes": MODES,
        "corruptions": CORRUPTIONS,
        "criteria": crit,
        "sources": src,
        "rs": rs,
        "ood_rs": ood_rs,
        "mixed_rs": mixed_rs,
        "shift_rs": shift_rs,
        "severities": sevs,
        "special": SPECIAL_METHODS,
        "colors": method_colors(),
    }


def build_html(results: Path) -> str:
    """Assemble the complete report page."""
    frames = read_frames(results)
    meta = build_meta(frames)
    main = compute_main_scores(frames, meta["rs"])
    print_main_scores(main)
    frames["derived/main_scores"] = main
    figs = collect_figures(results)
    cards = headline_cards(frames, main, meta["rs"])
    payload = {"frames": frames, "meta": meta}
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    title = html.escape(getattr(report_text, "TITLE", "SGR selective prediction results") if report_text else "SGR selective prediction results")
    subtitle = html.escape(getattr(report_text, "SUBTITLE", "") if report_text else "")
    commit = git_commit()
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    meta_line = f"Generated {stamp}" + (f" &middot; commit {html.escape(commit)}" if commit else "")

    toc = [(k, SECTION_TITLES[k]) for k in SECTION_KEYS] + [("appendix", "Appendix")]
    toc_html = "".join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t in toc)
    body = []
    for k in SECTION_KEYS:
        parts = [f'<section id="{k}"><h2>{html.escape(SECTION_TITLES[k])}</h2>']
        if k == "overview":
            parts.append('<div class="cards">' + "".join(f'<div class="card"><div class="cl">{html.escape(a)}</div><div class="cv">{html.escape(b)}</div><div class="cd">{html.escape(c)}</div></div>' for a, b, c in cards) + "</div>")
        parts.append(f'<div class="prose">{get_text(k)}</div>')
        parts.extend(figure_html(p) for p in figs.get(k, []))
        parts.extend(f'<div class="view" data-view="{v}"></div>' for v in SECTION_VIEWS.get(k, []))
        parts.append("</section>")
        body.append("".join(parts))
    app = ['<section id="appendix"><h2>Appendix</h2>', '<h3>Fixed-threshold baselines</h3><div class="view" data-view="fixed"></div>']
    if figs["appendix"]:
        app.append("<h3>Further figures</h3>")
        app.extend(figure_html(p) for p in figs["appendix"])
    app.append('<h3>Raw data</h3><p class="muted">Every csv in full (not printed in the PDF).</p><div class="view" data-view="raw"></div></section>')
    body.append("".join(app))

    light, dark = _VARS_LIGHT, _VARS_DARK
    css = _CSS.replace("__LIGHT__", light).replace("__DARK__", dark)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents">{toc_html}</nav>
<main>
<header class="top"><h1>{title}</h1><p class="sub">{subtitle}</p><p class="meta">{meta_line}</p></header>
{"".join(body)}
</main>
</div>
<div id="lb" class="lightbox" hidden><img alt=""></div>
<script type="application/json" id="data">{blob}</script>
<script>{_JS}</script>
</body>
</html>
"""


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
    """Print the report to PDF with headless Chrome; returns False if Chrome is missing or fails."""
    exe = find_chrome(chrome)
    if exe is None:
        print("Chrome not found, skipping the PDF (pass --chrome PATH).")
        return False
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)  # noqa: S603
    except (OSError, subprocess.SubprocessError) as e:
        print(f"PDF generation failed: {e}")
        return False
    return pdf_path.exists()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--out", type=Path, default=None, help="Output html path (default: <results>/report.html).")
    p.add_argument("--pdf", action="store_true", help="Also print <out>.pdf with headless Chrome.")
    p.add_argument("--chrome", default=None, help="Path of the Chrome/Chromium executable.")
    return p.parse_args()


def main() -> None:
    """Write the report (and the PDF with ``--pdf``)."""
    args = parse_args()
    out = args.out or args.results / "report.html"
    page = build_html(args.results)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if args.pdf:
        pdf = out.with_suffix(".pdf")
        if render_pdf(out, pdf, args.chrome):
            print(f"Wrote {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")


_VARS_LIGHT = (
    "--bg:#fbfaf7;--fg:#1d2330;--muted:#667085;--border:#dcdfe6;--card:#ffffff;--accent:#1a5fb4;"
    "--heat:#2e9c6a;--bad:#c62828;--th:#f0efe9;--soft:#f4f3ee;"
)
_VARS_DARK = (
    "--bg:#14171c;--fg:#e4e6ea;--muted:#9aa3b2;--border:#2d333d;--card:#1b1f26;--accent:#6ea8ff;"
    "--heat:#3fb37f;--bad:#ff7b72;--th:#20252d;--soft:#1f242c;"
)

_CSS = """
:root{__LIGHT__}
@media (prefers-color-scheme: dark){:root{__DARK__}}
*{box-sizing:border-box;-webkit-print-color-adjust:exact;print-color-adjust:exact}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--bg);color:var(--fg);font-family:'Fira Sans',system-ui,-apple-system,'Segoe UI',sans-serif;font-size:16px;line-height:1.6;font-weight:400}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);gap:40px;max-width:1560px;margin:0 auto;padding:0 24px}
main{min-width:0;padding-bottom:80px}
.toc{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:32px 0;display:flex;flex-direction:column;gap:2px;font-size:14px}
.toc a{color:var(--muted);text-decoration:none;padding:4px 10px;border-left:2px solid var(--border)}
.toc a:hover{color:var(--fg)}
.toc a.active{color:var(--accent);border-left-color:var(--accent);font-weight:500}
header.top{padding:48px 0 8px}
h1{font-size:2.1rem;line-height:1.2;font-weight:600;margin:0 0 8px;max-width:30em}
.sub{font-size:1.15rem;color:var(--muted);font-weight:300;margin:0 0 6px;max-width:55ch}
.meta{font-size:13px;color:var(--muted);margin:0}
section{padding-top:36px;margin-top:28px;border-top:1px solid var(--border)}
h2{font-size:1.5rem;font-weight:600;margin:0 0 14px}
h3{font-size:1.15rem;font-weight:600;margin:28px 0 8px}
h4{font-size:1rem;font-weight:600;margin:22px 0 6px}
.prose{max-width:75ch}
.prose p,.prose li{hyphens:auto}
.prose code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:.9em}
.prose table{border-collapse:collapse;font-size:14px;margin:12px 0}
.prose th,.prose td{border:1px solid var(--border);padding:4px 8px}
.todo{color:var(--bad);font-style:italic}
.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:8px 0 24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.cl{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.cv{font-size:1.9rem;font-weight:600;line-height:1.2;margin:4px 0;font-variant-numeric:tabular-nums}
.cd{font-size:13px;color:var(--muted);line-height:1.35}
figure.fig{margin:22px 0;background:#fff;color:#222;border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center}
figure.fig img{max-width:100%;height:auto;cursor:zoom-in;display:block;margin:0 auto}
figure.fig figcaption{font-size:12px;color:#667085;margin-top:4px}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.82);overflow:auto;z-index:100;cursor:zoom-out;padding:20px}
.lightbox[hidden]{display:none}
.lightbox img{background:#fff;max-width:none;display:block;margin:0 auto}
.view{margin:20px 0}
.controls{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin:10px 0 12px}
.cg{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cg>span{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.cg button{font:inherit;font-size:13px;background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:3px 10px;cursor:pointer}
.cg button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.ptitle{display:none}
.tw{margin:8px 0 18px}
.toolbar{display:flex;gap:14px;align-items:center;margin-bottom:6px;font-size:13px}
.toolbar input{font:inherit;font-size:13px;padding:3px 8px;border:1px solid var(--border);border-radius:6px;background:var(--card);color:var(--fg);width:200px;max-width:100%}
.toolbar a{color:var(--accent)}
.scroll{overflow:auto;max-height:76vh;border:1px solid var(--border);border-radius:8px;background:var(--card);max-width:100%}
table.t{border-collapse:separate;border-spacing:0;font-size:13px;font-variant-numeric:tabular-nums;width:max-content;min-width:100%}
.t th,.t td{padding:4px 10px;white-space:nowrap;border-bottom:1px solid var(--border);text-align:right}
.t th:first-child,.t td:first-child{text-align:left;position:sticky;left:0;background:var(--card);z-index:1;border-right:1px solid var(--border)}
.t thead th{position:sticky;top:0;background:var(--th);z-index:2;cursor:pointer;font-weight:500;user-select:none}
.t thead th:first-child{z-index:3}
.t thead th .ar{display:inline-block;width:1em;color:var(--accent)}
.t td.num{background:color-mix(in srgb,var(--heat) calc(var(--h,0)*100%),transparent)}
.t td.b{font-weight:700}
.t td.bad{color:var(--bad)}
.t td.bad::before{content:'! ';font-weight:700}
.t .sd{color:var(--muted);font-size:.88em;font-weight:400}
.t .na{color:var(--muted);opacity:.6}
.t .txt{text-align:left}
.badge{display:inline-block;margin-left:6px;font-size:10px;line-height:1.4;padding:0 5px;border-radius:8px;color:var(--bad);border:1px solid var(--bad);font-weight:500}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:baseline}
details{margin:10px 0}
summary{cursor:pointer;font-weight:500}
@media (max-width:900px){
 .layout{display:block;padding:0 16px}
 .toc{position:sticky;top:0;z-index:20;background:var(--bg);flex-direction:row;overflow-x:auto;max-height:none;padding:8px 0;border-bottom:1px solid var(--border);gap:0}
 .toc a{white-space:nowrap;border-left:0;border-bottom:2px solid transparent}
 .toc a.active{border-bottom-color:var(--accent)}
 header.top{padding-top:24px}
 h1{font-size:1.6rem}
 .prose table{display:block;overflow-x:auto;max-width:100%}
}
@page{size:A4;margin:12mm}
@media print{
 :root{__LIGHT__}
 body{font-size:10pt;background:#fff}
 .toc,.controls,.toolbar,.lightbox,.noprint,details.raw{display:none!important}
 .layout{display:block;padding:0;max-width:none}
 section{break-before:auto}
 h2,h3,h4{break-after:avoid}
 figure.fig,tr,.card{break-inside:avoid}
 .panel[hidden]:not(.noprint){display:block!important}
 .ptitle{display:block;font-size:9pt;font-weight:600;color:var(--muted);margin:10px 0 2px}
 .scroll{overflow:visible;max-height:none;border:0}
 table.t{width:100%;min-width:0;font-size:6.5pt}
 .t th,.t td{white-space:normal;padding:1px 3px}
 .t th:first-child,.t td:first-child,.t thead th{position:static}
 thead{display:table-header-group}
 .tw{margin:4px 0 10px}
 details::details-content{content-visibility:visible;display:block}
 .cards{grid-template-columns:repeat(5,1fr)}
 .cv{font-size:15pt}
}
"""

_JS = r"""
(function () {
'use strict';
const D = JSON.parse(document.getElementById('data').textContent);
const F = D.frames, M = D.meta;
for (const id in F) { const f = F[id]; f.idx = {}; f.cols.forEach((c, i) => { f.idx[c] = i; }); }
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const mean = c => (c == null ? null : (Array.isArray(c) ? c[0] : (typeof c === 'number' ? c : null)));
const methodOf = n => M.special[n] || String(n).split('_')[0];
const DOT_SRC = new Set(['criterion', 'source', 'method']);
const dot = n => '<i class="dot" style="background:' + (M.colors[methodOf(n)] || '#999') + '"></i>' + esc(n);

function fmtNum(x, f) { return f.pct ? (x * 100).toFixed(0) + '%' : x.toFixed(f.d); }
function fmtCell(c, f) {
  if (c == null || c === '') return '<span class="na">n/a</span>';
  if (typeof c === 'string') return esc(c);
  if (!f) return String(Array.isArray(c) ? c[0] : c);
  if (Array.isArray(c)) return fmtNum(c[0], f) + ' <span class="sd">&plusmn; ' + fmtNum(c[1], f) + '</span>';
  return fmtNum(c, f);
}
function getRows(fid, where) {
  const f = F[fid];
  const wi = Object.entries(where || {}).map(([k, v]) => [f.idx[k], String(v)]);
  return f.rows.filter(r => wi.every(([i, v]) => String(r[i]) === v));
}
function colsOf(fid, where, specs) {
  return specs.map(s => ({ fid: fid, where: where, src: s[0], label: s[1] == null ? s[0] : s[1], badge: s[2] }));
}
function allCols(fid) { return colsOf(fid, {}, F[fid].cols.map(c => [c, c])); }

function buildTable(spec) {
  const cols = spec.cols;
  const keyed = !!spec.keys;
  const cache = new Map();
  const rowMap = c => {
    const k = c.fid + '|' + JSON.stringify(c.where || {});
    if (!cache.has(k)) {
      const f = F[c.fid], ki = f.idx[spec.keyCol || 'criterion'], m = new Map();
      getRows(c.fid, c.where).forEach(r => m.set(r[ki], r));
      cache.set(k, m);
    }
    return cache.get(k);
  };
  let lines;
  if (keyed) {
    lines = spec.keys.map(key => ({
      key: key,
      cells: cols.map(c => { const r = rowMap(c).get(key); return r ? r[F[c.fid].idx[c.src]] : null; }),
      badges: cols.map(c => { if (!c.badge) return null; const r = rowMap(c).get(key); return r ? r[F[c.fid].idx[c.badge]] : null; }),
    }));
  } else {
    lines = getRows(cols[0].fid, spec.where).map(r => ({
      key: null, cells: cols.map(c => r[F[c.fid].idx[c.src]]), badges: cols.map(() => null),
    }));
  }
  const fmts = cols.map(c => c.fmt !== undefined ? c.fmt : F[c.fid].fmts[F[c.fid].idx[c.src]]);
  const nc = cols.length;
  // heat and best per column
  const heat = cols.map(() => null), best = cols.map(() => null);
  cols.forEach((c, j) => {
    const f = fmts[j];
    if (!f || !f.dir || spec.heat === false || F[c.fid].heat === false) return;
    const vals = lines.map(l => mean(l.cells[j])).filter(v => v != null);
    if (vals.length < 2) return;
    const s = vals.slice().sort((a, b) => a - b);
    if (s[0] === s[s.length - 1]) return;
    const lo = s.filter(v => v < 0).length;
    best[j] = f.dir < 0 ? s[0] : s[s.length - 1];
    heat[j] = v => {
      const below = s.filter(x => x < v).length, eq = s.filter(x => x === v).length;
      const p = (below + (eq - 1) / 2) / (s.length - 1);
      return 0.38 * (f.dir < 0 ? 1 - p : p);
    };
  });
  const wrap = el('div', 'tw');
  const bar = el('div', 'toolbar');
  const inp = el('input'); inp.type = 'search'; inp.placeholder = 'filter rows';
  bar.appendChild(inp);
  const csvA = el('a', null, 'download CSV'); csvA.download = (spec.csvName || spec.cols[0].fid.replace(/[^\w.-]+/g, '_')) + '.csv'; csvA.href = '#';
  const setHref = () => { if (csvA.href.indexOf('data:') !== 0) csvA.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(F[spec.cols[0].fid].csv); };
  ['pointerenter', 'focus', 'touchstart', 'click'].forEach(ev => csvA.addEventListener(ev, setHref));
  bar.appendChild(csvA);
  wrap.appendChild(bar);
  const sc = el('div', 'scroll');
  const tbl = el('table', 't');
  const heads = [];
  if (keyed) heads.push(spec.keyLabel || 'criterion');
  cols.forEach(c => heads.push(c.label));
  const thead = el('thead'); const tr0 = el('tr');
  heads.forEach((h, i) => { const th = el('th', null, esc(h) + '<span class="ar"></span>'); th.dataset.i = i; tr0.appendChild(th); });
  thead.appendChild(tr0); tbl.appendChild(thead);
  const tbody = el('tbody');
  const off = keyed ? 1 : 0;
  const sortVals = [];
  const html = lines.map((l, li) => {
    const sv = [];
    let h = '<tr>';
    if (keyed) { h += '<td>' + (spec.noDot ? esc(l.key) : dot(l.key)) + '</td>'; sv.push(String(l.key)); }
    l.cells.forEach((c, j) => {
      const f = fmts[j];
      const m = mean(c);
      const isText = typeof c === 'string';
      let cls = '', st = '';
      if (isText || (c == null && !f)) cls = 'txt';
      else cls = 'num';
      if (m != null && heat[j]) st = ' style="--h:' + heat[j](m).toFixed(3) + '"';
      if (m != null && best[j] != null && m === best[j]) cls += ' b';
      if (m != null && f && f.viol && m > 0.01) cls += ' bad';
      let inner;
      if (isText && DOT_SRC.has(cols[j].src)) inner = dot(c); else inner = fmtCell(c, f);
      const bd = l.badges[j];
      if (bd != null && mean(bd) != null && mean(bd) > 0.01) inner += '<span class="badge">viol ' + (mean(bd) * 100).toFixed(0) + '%</span>';
      h += '<td class="' + cls + '"' + st + '>' + inner + '</td>';
      sv.push(isText ? c : (m == null ? null : m));
    });
    sortVals.push(sv);
    return h + '</tr>';
  }).join('');
  tbody.innerHTML = html;
  const trs = Array.from(tbody.children);
  const items = trs.map((tr, i) => ({ tr: tr, v: sortVals[i], i: i, text: tr.textContent.toLowerCase() }));
  tbl.appendChild(tbody); sc.appendChild(tbl); wrap.appendChild(sc);
  let st = { i: -1, dir: 0 };
  const apply = () => {
    const arr = items.slice();
    if (st.dir) arr.sort((a, b) => {
      const x = a.v[st.i], y = b.v[st.i];
      if (x == null && y == null) return a.i - b.i;
      if (x == null) return 1;
      if (y == null) return -1;
      const c = (typeof x === 'string' || typeof y === 'string') ? String(x).localeCompare(String(y), undefined, { numeric: true }) : x - y;
      return c ? c * st.dir : a.i - b.i;
    }); else arr.sort((a, b) => a.i - b.i);
    arr.forEach(o => tbody.appendChild(o.tr));
    tr0.querySelectorAll('.ar').forEach((a, i) => { a.textContent = i === st.i ? (st.dir > 0 ? '\u25B2' : '\u25BC') : ''; });
  };
  tr0.addEventListener('click', e => {
    const th = e.target.closest('th'); if (!th) return;
    const i = +th.dataset.i;
    const k = keyed ? i : i;
    if (st.i === k) st.dir = st.dir === 1 ? -1 : (st.dir === -1 ? 0 : 1); else { st.i = k; st.dir = 1; }
    // sort values array: keyed has key at index 0, same layout as heads
    apply();
  });
  if (spec.sort) { st = { i: spec.sort.i + off, dir: spec.sort.dir }; apply(); }
  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    items.forEach(o => { o.tr.hidden = q !== '' && o.text.indexOf(q) < 0; });
  });
  return wrap;
}

function product(lists) { return lists.reduce((acc, l) => acc.flatMap(a => l.map(x => a.concat([x]))), [[]]); }

function tabbed(root, dims, make) {
  const bar = el('div', 'controls'), panels = el('div', 'panels');
  const sel = {};
  dims.forEach(d => { sel[d.key] = d.values[d.def || 0][0]; });
  const defs = {}; dims.forEach(d => { defs[d.key] = d.values[d.def || 0][0]; });
  const btns = [];
  dims.forEach(d => {
    const g = el('div', 'cg'); g.appendChild(el('span', null, esc(d.label)));
    d.values.forEach(v => {
      const b = el('button', null, esc(v[1])); b.type = 'button';
      b.addEventListener('click', () => { sel[d.key] = v[0]; update(); });
      b.dataset.k = d.key; b.dataset.v = v[0]; g.appendChild(b); btns.push(b);
    });
    bar.appendChild(g);
  });
  const ps = [];
  product(dims.map(d => d.values)).forEach(combo => {
    const s = {}; dims.forEach((d, i) => { s[d.key] = combo[i][0]; });
    const p = el('div', 'panel');
    p.appendChild(el('div', 'ptitle', esc(combo.map(c => c[1]).join(' / '))));
    p.appendChild(make(s));
    if (dims.some(d => !d.printAll && s[d.key] !== defs[d.key])) p.classList.add('noprint');
    ps.push([s, p]); panels.appendChild(p);
  });
  function update() {
    btns.forEach(b => b.classList.toggle('on', sel[b.dataset.k] === b.dataset.v));
    ps.forEach(([s, p]) => { p.hidden = !dims.every(d => s[d.key] === sel[d.key]); });
  }
  update();
  root.appendChild(bar); root.appendChild(panels);
}
const pairs = a => a.map(x => [String(x), String(x)]);
const rTabs = (rs, label) => ({ key: 'r', label: label || 'r*', values: rs.map(r => [String(r), 'r* = ' + r]), printAll: true });
const CRIT = M.criteria, MODES = M.modes, CORR = M.corruptions;
const h = (t, tag) => el(tag || 'h3', null, esc(t));
const note = t => el('p', 'muted', t);

const VIEWS = {};
VIEWS.repro = root => {
  [['table', 'Table 1 reproduction: held-out risk and coverage'], ['table_sgr', 'SGR rule'], ['table_coverage', 'Coverage-matched thresholds']].forEach(([id, t]) => {
    root.appendChild(h(t)); root.appendChild(buildTable({ cols: allCols(id), heat: false }));
  });
};
VIEWS.clean = root => {
  root.appendChild(h('Clean-data ranking of all criteria'));
  root.appendChild(buildTable({ keys: CRIT, cols: colsOf('shift/id_aurc', {}, [['AURC x1000', 'AURC x 1000'], ['E-AURC x1000', 'E-AURC x 1000'], ['accuracy', 'accuracy']]), sort: { i: 0, dir: 1 } }));
};
VIEWS.main = root => {
  root.appendChild(h('Main score per method'));
  root.appendChild(note('Main score = criterion with the lowest clean AURC within the method. Values are means over seeds and splits; the shift columns average the four corruptions.'));
  root.appendChild(buildTable({ cols: allCols('derived/main_scores'), sort: { i: 2, dir: 1 }, csvName: 'main_scores' }));
};
VIEWS.rules = root => {
  root.appendChild(h('Test coverage, risk and violation share per rule'));
  root.appendChild(note('Rows are criteria, columns the accept/abstain modes. Coverage cells carry a badge when the violation share (test risk above r*) exceeds 1%.'));
  const metrics = [['cov', 'coverage'], ['risk', 'risk'], ['viol', 'violation']];
  tabbed(root, [rTabs(M.rs), { key: 'm', label: 'metric', values: metrics, printAll: true }], s => buildTable({
    keys: CRIT, cols: colsOf('options/id_r' + s.r, {}, MODES.map(m => [m + ' ' + s.m, m, s.m === 'cov' ? m + ' viol' : null])),
  }));
};
VIEWS.cert = root => {
  root.appendChild(h('SGR certification check'));
  root.appendChild(note('Share of splits in which the SGR search could not certify any threshold (uncertified), per r*, and thresholds that differ from the reference implementation.'));
  root.appendChild(buildTable({ keys: CRIT, cols: colsOf('shift/sgr_check', {}, [['thresholds differing from reference', 'differs from reference'], ...M.rs.map(r => ['uncertified r*=' + r, 'uncertified r*=' + r])]) }));
  root.appendChild(h('Empirical, SGR and coverage-matched thresholds'));
  tabbed(root, [rTabs(M.rs)], s => buildTable({
    keys: CRIT, cols: colsOf('shift/id_thresholds', { 'r*': s.r }, ['emp', 'sgr', 'cov'].flatMap(m => [[m + ' risk', m + ' risk'], [m + ' cov', m + ' cov'], [m + ' viol', m + ' viol']]).concat([['sgr bound', 'sgr bound']])),
  }));
};
VIEWS.ood = root => {
  tabbed(root, [{ key: 'ds', label: 'OOD set', values: [['svhn', 'SVHN'], ['cifar100', 'CIFAR-100']], printAll: true }], s => {
    const box = el('div');
    box.appendChild(h('AUROC and accepted OOD share at the clean thresholds', 'h4'));
    const specs = [['AUROC', 'AUROC']];
    ['emp', 'sgr', 'cov'].forEach(m => M.ood_rs.forEach(r => specs.push([m + ' accepted r*=' + r, m + ' r*=' + r])));
    box.appendChild(buildTable({ keys: CRIT, cols: colsOf('shift/ood_' + s.ds, {}, specs) }));
    box.appendChild(h('Accepted OOD share for all modes', 'h4'));
    tabbed(box, [rTabs(M.ood_rs)], t => buildTable({ keys: CRIT, cols: colsOf('options/ood_' + s.ds, {}, MODES.map(m => [m + ' r*=' + t.r, m])) }));
    const det = el('details'); det.appendChild(el('summary', null, 'Mixed ID + OOD test set'));
    tabbed(det, [rTabs(M.mixed_rs)], t => buildTable({
      keys: CRIT, cols: colsOf('shift/mixed_' + s.ds, { 'r*': t.r }, ['emp', 'sgr', 'cov'].flatMap(m => [[m + ' cov', m + ' cov'], [m + ' risk', m + ' risk']])),
    }));
    box.appendChild(det);
    return box;
  });
};
const corrTabs = () => ({ key: 'c', label: 'corruption', values: CORR.map(c => [c, c.replace('_', ' ')]), printAll: true });
VIEWS.shift = root => {
  root.appendChild(h('Test risk and coverage under covariate shift'));
  root.appendChild(note('Thresholds come from clean data. Printed version: r* = 0.03, SGR mode.'));
  const rs = M.shift_rs, di = Math.max(0, rs.indexOf('0.03'));
  tabbed(root, [
    corrTabs(),
    { key: 'r', label: 'r*', values: rs.map(r => [r, r]), def: di },
    { key: 'mode', label: 'mode', values: [['emp', 'emp'], ['sgr', 'sgr'], ['cov', 'cov']], def: 1 },
    { key: 'm', label: 'metric', values: [['risk', 'risk'], ['cov', 'coverage']], printAll: true },
  ], s => {
    const cols = [];
    const sev = ['0'].concat(M.severities.filter(x => x !== '0'));
    sev.forEach(v => cols.push(v === '0' ? { fid: 'shift/id_aurc', where: {}, src: 'accuracy', label: 'accuracy clean' } : { fid: 'shift/shift_' + s.c, where: { severity: v }, src: 'accuracy', label: 'accuracy sev ' + v }));
    sev.forEach(v => cols.push(v === '0' ? { fid: 'shift/id_thresholds', where: { 'r*': s.r }, src: s.mode + ' ' + s.m, label: s.m + ' clean' } : { fid: 'shift/shift_' + s.c, where: { severity: v }, src: 'r*=' + s.r + ' ' + s.mode + ' ' + s.m, label: s.m + ' sev ' + v }));
    return buildTable({ keys: CRIT, cols: cols, csvName: 'shift_' + s.c });
  });
};
VIEWS.shiftopt = root => {
  root.appendChild(h('All modes under covariate shift'));
  const rs = M.shift_rs, di = Math.max(0, rs.indexOf('0.03'));
  const sv = M.severities.filter(x => x !== '0');
  tabbed(root, [
    corrTabs(),
    { key: 'r', label: 'r*', values: rs.map(r => [r, r]), def: di },
    { key: 'sev', label: 'severity', values: sv.map(v => [v, v]), def: sv.length - 1 },
    { key: 'm', label: 'metric', values: [['risk', 'risk'], ['cov', 'coverage']], printAll: true },
  ], s => buildTable({ keys: CRIT, cols: colsOf('options/shift_' + s.c + '_r' + s.r, { severity: s.sev }, MODES.map(m => [m + ' ' + s.m, m])) }));
};
VIEWS.calib = root => {
  root.appendChild(h('Clean calibration'));
  const cc = ['accuracy', 'T', 'ECE raw', 'ECE TS', 'NLL raw', 'NLL TS', 'Brier raw', 'Brier TS'];
  root.appendChild(buildTable({ keys: M.sources, keyCol: 'source', keyLabel: 'source', cols: colsOf('calibration/clean_calibration', {}, cc.map(c => [c, c])) }));
  [['calibration/ranking_aurc', 'AURC x 1000 of the ranking scores, raw vs temperature scaled'], ['calibration/ranking_eaurc', 'E-AURC x 1000 of the ranking scores']].forEach(([id, t]) => {
    root.appendChild(h(t));
    root.appendChild(buildTable({ keys: M.sources, keyCol: 'source', keyLabel: 'source', cols: colsOf(id, {}, F[id].cols.filter(c => c !== 'source').map(c => [c, c])) }));
  });
  root.appendChild(h('Calibration under covariate shift'));
  tabbed(root, [corrTabs(), { key: 'k', label: 'metric', values: [['ECE', 'ECE'], ['NLL', 'NLL'], ['Brier', 'Brier']], printAll: true }], s => {
    const cols = [];
    ['raw', 'TS'].forEach(v => ['0'].concat(M.severities.filter(x => x !== '0')).forEach(sv => cols.push({ fid: 'calibration/shift_' + s.c, where: { severity: sv }, src: s.k + ' ' + v, label: v + ' sev ' + sv })));
    return buildTable({ keys: M.sources, keyCol: 'source', keyLabel: 'source', cols: cols, csvName: 'calibration_shift_' + s.c });
  });
};
VIEWS.fixed = root => { root.appendChild(buildTable({ cols: allCols('fixed_threshold'), heat: false })); };
VIEWS.raw = root => {
  Object.keys(F).filter(id => id.indexOf('derived/') !== 0).forEach(id => {
    const d = el('details', 'raw');
    d.appendChild(el('summary', null, esc(id) + '.csv <span class="muted">(' + F[id].rows.length + ' rows, ' + F[id].cols.length + ' columns)</span>'));
    let built = false;
    d.addEventListener('toggle', () => { if (d.open && !built) { built = true; d.appendChild(buildTable({ cols: allCols(id), heat: false })); } });
    root.appendChild(d);
  });
};

document.querySelectorAll('[data-view]').forEach(n => {
  try { VIEWS[n.dataset.view](n); n.dataset.ok = '1'; }
  catch (e) { console.error(n.dataset.view, e); n.appendChild(el('p', 'todo', 'view failed: ' + esc(e.message))); }
});

// lightbox
const lb = document.getElementById('lb');
document.addEventListener('click', e => {
  const img = e.target.closest && e.target.closest('figure.fig img');
  if (img) { lb.querySelector('img').src = img.src; lb.hidden = false; return; }
  if (!lb.hidden) lb.hidden = true;
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.hidden = true; });

// open collapsibles before printing
let reopened = [];
window.addEventListener('beforeprint', () => { reopened = Array.from(document.querySelectorAll('details:not(.raw):not([open])')); reopened.forEach(d => { d.open = true; }); });
window.addEventListener('afterprint', () => { reopened.forEach(d => { d.open = false; }); reopened = []; });

// scroll spy
const links = Array.from(document.querySelectorAll('.toc a'));
if ('IntersectionObserver' in window) {
  const io = new IntersectionObserver(es => {
    es.forEach(en => { if (en.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id)); });
  }, { rootMargin: '-10% 0px -80% 0px' });
  document.querySelectorAll('main section').forEach(s => io.observe(s));
}
})();
"""


if __name__ == "__main__":
    main()
