"""Matched comparisons at SGR's operating point (E3) and scores per method family (E4) on clean CIFAR-10.

Protocol of ``paper_table.py``: per seed unit, ``n_splits`` permutations of the 10k test set (one list shared by all
criteria); the first half selects, the second half is the test half. SGR (probly's ``SGRSelector``, Algorithm 1 of
Geifman and El-Yaniv 2017, arXiv 1705.08500) is fitted at each desired risk r*.

E3: a reference criterion (default ``dropout_scratch_sr``) is fitted with SGR on its selection half. On every split where
it certifies, SGR fixes an operating point: its selection-half coverage and its realized test risk. Every other criterion
is then compared at that point, in two ways. ``risk_at_cov``: a label-free ``CoverageSelector`` is calibrated to the
selection-half coverage of the reference on the criterion's own selection half, and its test risk is recorded.
``cov_at_risk``: the largest test coverage of the criterion whose risk is at most the reference's realized test risk.
Differences to the reference are paired over the certified (unit, split) pairs.

E4: for the method families with several scores, the same two quantities (at the reference's operating point) and the
clean AURC and AUGRC, each against the family's own ``maxprob`` (SR) score.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

from evaluate import THIN_FONT, setup_fonts
from evaluate_shift import CRITERIA, METHOD_KEYS, load_dataset, sgr_selector_threshold
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.selective_prediction import CoverageSelector  # noqa: E402
from sgr_experiment.metrics import apply_threshold, augrc, aurc, coverage_at_risk  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

R_GRID = [0.01, 0.02, 0.03, 0.05]
FALLBACK_REFERENCE = "sr_base"
# Families with several scores: the baseline is ``{family}_maxprob``; the other scores are compared against it.
FAMILY_SCORES = {
    "mc": ["maxprob", "total", "aleatoric", "epistemic", "variance"],
    "ens": ["maxprob", "total", "aleatoric", "epistemic"],
    "dropout_scratch": ["sr", "maxprob", "total", "aleatoric", "epistemic"],
    "swag": METHOD_KEYS["swag"],
    "laplace": METHOD_KEYS["laplace"],
    "vbll": METHOD_KEYS["vbll"],
}
VERDICT_RISKS = (0.01, 0.03)
PLOT_RISK = 0.03
MAIN_PREFIXES = ("sr_", "ens_", "dropout_scratch_")
MAIN_EXACT = ("mc_maxprob", "ddu_maxprob", "laplace_maxprob")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "matched")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of the SGR bound.")
    p.add_argument("--reference", default="dropout_scratch_sr", help="Reference criterion (falls back to sr_base).")
    return p.parse_args()


def nan_array(*shape: int) -> np.ndarray:
    """Array of NaN."""
    return np.full(shape, np.nan)


def reference_context(units: list, labels: np.ndarray, splits: list[np.ndarray], delta: float) -> dict:
    """SGR operating points of the reference: certified flag, selection coverage, test risk and coverage.

    Every array has shape ``(n_units, n_splits)`` per r*.
    """
    half = len(labels) // 2
    n_u, n_s = len(units), len(splits)
    ctx = {k: {r: nan_array(n_u, n_s) for r in R_GRID} for k in ("cert", "c_sel", "r_sgr", "c_sgr")}
    for u, (crit, pred) in enumerate(units):
        loss = (pred != labels).astype(np.float64)
        for s, perm in enumerate(splits):
            sel, test = perm[:half], perm[half:]
            for r in R_GRID:
                thr, _ = sgr_selector_threshold(crit[sel], loss[sel], r, delta)
                risk, cov = apply_threshold(crit[test], loss[test], thr)
                certified = thr != -np.inf and np.isfinite(risk)
                ctx["cert"][r][u, s] = float(certified)
                if certified:
                    ctx["c_sel"][r][u, s] = apply_threshold(crit[sel], loss[sel], thr)[1]
                    ctx["r_sgr"][r][u, s] = risk
                    ctx["c_sgr"][r][u, s] = cov
    return ctx


def evaluate_criterion(units: list, n_ref: int, labels: np.ndarray, splits: list[np.ndarray], ctx: dict, delta: float) -> dict:
    """Matched quantities and own SGR/AURC of one criterion.

    Reference unit ``u`` is paired with unit ``u`` of the criterion if it has as many units as the reference, else with
    unit 0. Matched arrays have shape ``(n_ref, n_splits)`` per r* and are NaN off the certified pairs; own arrays have
    shape ``(n_units, n_splits)``.
    """
    half = len(labels) // 2
    n_x, n_s = len(units), len(splits)
    res = {k: {r: nan_array(n_ref, n_s) for r in R_GRID} for k in ("rac", "car", "ach")}
    res["own_cov"] = {r: nan_array(n_x, n_s) for r in R_GRID}
    res["own_cert"] = {r: nan_array(n_x, n_s) for r in R_GRID}
    res["aurc"] = nan_array(n_x, n_s)
    res["augrc"] = nan_array(n_x, n_s)
    losses = [(pred != labels).astype(np.float64) for _, pred in units]
    for xu, (crit, _) in enumerate(units):
        for s, perm in enumerate(splits):
            sel, test = perm[:half], perm[half:]
            res["aurc"][xu, s] = aurc(crit[test], losses[xu][test])
            res["augrc"][xu, s] = augrc(crit[test], losses[xu][test])
            for r in R_GRID:
                thr, _ = sgr_selector_threshold(crit[sel], losses[xu][sel], r, delta)
                certified = thr != -np.inf
                res["own_cert"][r][xu, s] = float(certified)
                res["own_cov"][r][xu, s] = apply_threshold(crit[test], losses[xu][test], thr)[1] if certified else 0.0
    for u in range(n_ref):
        xu = u if n_x == n_ref else 0
        crit, loss = units[xu][0], losses[xu]
        for s, perm in enumerate(splits):
            sel, test = perm[:half], perm[half:]
            for r in R_GRID:
                if not ctx["cert"][r][u, s]:
                    continue
                c_sel = ctx["c_sel"][r][u, s]
                if c_sel > 0:
                    thr = CoverageSelector(float(c_sel)).calibrate(crit[sel]).threshold
                    risk, cov = apply_threshold(crit[test], loss[test], thr)
                    res["rac"][r][u, s] = risk
                    res["ach"][r][u, s] = cov
                res["car"][r][u, s] = coverage_at_risk(crit[test], loss[test], ctx["r_sgr"][r][u, s])
    return res


def stats(values: np.ndarray) -> tuple[float, float]:
    """Mean and std over the finite values (NaN if there are none)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def paired(a: np.ndarray, b: np.ndarray, lower_is_better: bool) -> tuple[float, float, float, int]:
    """Mean, std and share of strictly better pairs of ``a - b`` over the pairs finite in both."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    better = (d < 0) if lower_is_better else (d > 0)
    return float(d.mean()), float(d.std()), float(better.mean()), int(d.size)


def ms(mean: float, std: float, digits: int = 4) -> str:
    """Markdown cell ``mean +- std`` (``n/a`` if there is no value)."""
    return "n/a" if not np.isfinite(mean) else f"{mean:.{digits}f} +- {std:.{digits}f}"


def pct(x: float) -> str:
    """Share as a percentage (``n/a`` if there is no value)."""
    return "n/a" if not np.isfinite(x) else f"{x * 100:.0f}%"


def e3_rows(res: dict, active: list[str], ref: str) -> dict[tuple[str, float], dict]:
    """E3 statistics per ``(criterion, r*)``."""
    rows = {}
    for c in active:
        for r in R_GRID:
            row = {}
            row["own_cov"], row["own_cov_std"] = stats(res[c]["own_cov"][r])
            row["own_cert"] = stats(res[c]["own_cert"][r])[0]
            row["rac"], row["rac_std"] = stats(res[c]["rac"][r])
            row["car"], row["car_std"] = stats(res[c]["car"][r])
            row["ach"], _ = stats(res[c]["ach"][r])
            row["d_rac"], row["d_rac_std"], row["b_rac"], row["n"] = paired(res[c]["rac"][r], res[ref]["rac"][r], True)
            row["d_car"], row["d_car_std"], row["b_car"], _ = paired(res[c]["car"][r], res[ref]["car"][r], False)
            rows[c, r] = row
    return rows


def family_members(active: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Families present in ``active``: baseline criterion and the scores shown (baseline first)."""
    fams = {}
    for fam, keys in FAMILY_SCORES.items():
        base = f"{fam}_maxprob"
        if base not in active:
            continue
        scores = [f"{fam}_{k}" for k in keys if f"{fam}_{k}" in active]
        scores.sort(key=lambda c: (c != base, c != f"{fam}_sr"))
        fams[fam] = (base, scores)
    return fams


def e4_rows(res: dict, fams: dict[str, tuple[str, list[str]]]) -> dict[tuple[str, float], dict]:
    """E4 statistics per ``(score, r*)``, deltas against the family's maxprob, plus the verdict per score."""
    rows = {}
    for fam, (base, scores) in fams.items():
        base_aurc = stats(res[base]["aurc"] * 1000)[0]
        for c in scores:
            a_mean, a_std = stats(res[c]["aurc"] * 1000)
            g_mean, g_std = stats(res[c]["augrc"] * 1000)
            beats = True
            for r in R_GRID:
                row = {"family": fam, "aurc": a_mean, "aurc_std": a_std, "augrc": g_mean, "augrc_std": g_std}
                row["d_rac"], row["d_rac_std"], row["b_rac"], row["n"] = paired(res[c]["rac"][r], res[base]["rac"][r], True)
                row["d_car"], row["d_car_std"], row["b_car"], _ = paired(res[c]["car"][r], res[base]["car"][r], False)
                rows[c, r] = row
                if r in VERDICT_RISKS and not (row["d_rac"] < 0 and row["b_rac"] > 0.5):
                    beats = False
            is_baseline = c in (base, f"{fam}_sr")
            verdict = beats and a_mean < base_aurc and not is_baseline
            for r in R_GRID:
                rows[c, r]["beats"] = verdict
    return rows


def build_markdown(args: argparse.Namespace, ref: str, ref_note: str, ctx: dict, active: list[str], e3: dict, fams: dict, e4: dict) -> str:
    """Markdown report with the E3 table per r* and the E4 table per family."""
    lines = [
        "# Matched comparisons at SGR's operating point (clean CIFAR-10)",
        "",
        f"Seeds {' '.join(map(str, args.seeds))} x {args.n_splits} selection/test splits (5k/5k), delta {args.delta}. "
        f"Reference criterion: `{ref}`{ref_note}. On every (unit, split) pair where SGR (Geifman and El-Yaniv 2017) "
        "certifies the reference at r*, its selection-half coverage and realized test risk define the operating point. "
        "`risk @ ref cov`: test risk of a CoverageSelector calibrated (label-free) to the reference's selection-half "
        "coverage on the criterion's own selection half. `cov @ ref risk`: largest test coverage of the criterion at "
        "risk <= the reference's realized test risk. `delta` is the paired difference to the reference (criterion minus "
        "reference) as mean +- std over the certified pairs; `better` is the share of pairs where the criterion is "
        "strictly better (lower risk, higher coverage). `own SGR cov` and `certified` refer to the criterion's own SGR "
        "on all splits (coverage 0 if it does not certify). The deep ensemble has one unit and is paired with unit 0 of "
        "every reference unit.",
        "",
        "# E3: matched comparison",
        "",
    ]
    for r in R_GRID:
        cert = ctx["cert"][r]
        lines += [
            f"## r* = {r:.2f}",
            "",
            f"Reference `{ref}` certified on {pct(float(cert.mean()))} of the splits ({int(cert.sum())} of {cert.size} pairs).",
            "",
            "| criterion | own SGR cov | certified | risk @ ref cov | delta | better | cov @ ref risk | delta | better |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in active:
            x = e3[c, r]
            name = f"{CRITERIA[c]['label']} (`{c}`)" + (" (ref)" if c == ref else "")
            lines.append(
                f"| {name} | {ms(x['own_cov'], x['own_cov_std'])} | {pct(x['own_cert'])} | {ms(x['rac'], x['rac_std'])} "
                f"| {ms(x['d_rac'], x['d_rac_std'])} | {pct(x['b_rac'])} | {ms(x['car'], x['car_std'])} "
                f"| {ms(x['d_car'], x['d_car_std'])} | {pct(x['b_car'])} |"
            )
        lines.append("")
    lines += [
        "# E4: scores per family",
        "",
        "Per family, every score against the family's `maxprob` (SR) score, at the reference's operating points above "
        "(same certified pairs as E3). AURC x1000 and AUGRC x1000 (Traub et al., 2024) are the mean +- std over unit x split test halves; the gate uses AURC only. A score beats SR if "
        f"its mean delta risk @ ref cov is below 0 and it is better in more than half of the pairs at both r* "
        f"{VERDICT_RISKS[0]} and {VERDICT_RISKS[1]}, and its AURC is lower "
        "than the baseline's. Gate rule: a score is kept if it beats SR for at least one family on E3.",
        "",
    ]
    kept = []
    for fam, (base, scores) in fams.items():
        lines += [
            f"## Family `{fam}` (baseline `{base}`)",
            "",
            "| score | AURC x1000 | AUGRC x1000 | "
            + " | ".join(f"d risk r*={r:.2f} | better | d cov r*={r:.2f} | better" for r in VERDICT_RISKS)
            + " | beats SR |",
            "|---|---|---|" + "---|---|---|---|" * len(VERDICT_RISKS) + "---|",
        ]
        for c in scores:
            x = e4[c, VERDICT_RISKS[0]]
            cells = []
            for r in VERDICT_RISKS:
                y = e4[c, r]
                cells += [ms(y["d_rac"], y["d_rac_std"]), pct(y["b_rac"]), ms(y["d_car"], y["d_car_std"]), pct(y["b_car"])]
            is_base = c == base or c == f"{fam}_sr"
            verdict = "baseline" if is_base else ("yes" if x["beats"] else "no")
            lines.append(f"| `{c}` | {ms(x['aurc'], x['aurc_std'], 3)} | {ms(x['augrc'], x['augrc_std'], 3)} | " + " | ".join(cells) + f" | {verdict} |")
            if x["beats"]:
                kept.append(c)
        winners = [c for c in scores if e4[c, VERDICT_RISKS[0]]["beats"]]
        lines += ["", f"Verdict: {', '.join(f'`{c}`' for c in winners) + ' beat(s) SR.' if winners else 'no score beats SR.'}", ""]
    lines += ["## Gate", "", f"Kept: {', '.join(f'`{c}`' for c in kept) if kept else 'none'}.", ""]
    return "\n".join(lines)


def write_csvs(out: Path, ref: str, active: list[str], e3: dict, e4: dict) -> None:
    """Long-format ``e3.csv`` and ``e4.csv``."""
    f3 = ["own_cov", "own_cert", "rac", "ach", "car", "d_rac", "b_rac", "d_car", "b_car"]
    stds = {"own_cov", "rac", "car", "d_rac", "d_car"}
    head3 = ["reference", "criterion", "r_star"] + [h for k in f3 for h in ((f"{k}_mean", f"{k}_std") if k in stds else (k,))] + ["n_pairs"]
    with (out / "e3.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head3)
        for c in active:
            for r in R_GRID:
                x = e3[c, r]
                vals = []
                for k in f3:
                    vals.append(x[k])
                    if k in stds:
                        vals.append(x[f"{k}_std"])
                w.writerow([ref, c, r, *vals, x["n"]])
    with (out / "e4.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "score", "r_star", "aurc_x1000_mean", "aurc_x1000_std", "augrc_x1000_mean", "augrc_x1000_std", "d_rac_mean", "d_rac_std", "better_rac",
                    "d_car_mean", "d_car_std", "better_car", "n_pairs", "beats_sr"])
        for (c, r), x in e4.items():
            w.writerow([x["family"], c, r, x["aurc"], x["aurc_std"], x["augrc"], x["augrc_std"], x["d_rac"], x["d_rac_std"], x["b_rac"],
                        x["d_car"], x["d_car_std"], x["b_car"], x["n"], int(x["beats"])])


def style_axes(ax: plt.Axes, has_fira: bool) -> None:
    """Semibold axis labels and thin tick labels."""
    ax.xaxis.label.set_fontweight("semibold")
    ax.yaxis.label.set_fontweight("semibold")
    if has_fira:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(THIN_FONT)


def save(fig: plt.Figure, path: Path) -> None:
    """Save without font warnings."""
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_e3(path: Path, e3: dict, active: list[str], ref: str) -> None:
    """Risk at the reference's coverage per main criterion and r*, with the reference's risk as a dashed line."""
    has_fira = setup_fonts()
    main = [c for c in active if c.startswith(MAIN_PREFIXES) or c in MAIN_EXACT]
    fig, axes = plt.subplots(1, len(R_GRID), figsize=(3.6 * len(R_GRID), 0.38 * len(main) + 1.6), sharey=True)
    ys = np.arange(len(main))
    for ax, r in zip(axes, R_GRID, strict=True):
        means = np.array([e3[c, r]["rac"] for c in main])
        stds = np.array([e3[c, r]["rac_std"] for c in main])
        ok = np.isfinite(means)
        ax.barh(ys[ok], means[ok], xerr=stds[ok], color=[CRITERIA[c]["color"] for c, k in zip(main, ok, strict=True) if k], alpha=0.85,
                error_kw={"lw": 1, "capsize": 2})
        ref_val = e3[ref, r]["rac"]
        if np.isfinite(ref_val):
            ax.axvline(ref_val, color="black", ls="--", lw=1, label=f"reference ({ref})")
        elif not ok.any():
            ax.text(0.5, 0.5, "n/a (nothing certified)", transform=ax.transAxes, ha="center")
        ax.set_title(f"r* = {r:.2f}")
        ax.set_xlabel("Test risk at reference coverage")
        ax.set_yticks(ys)
        ax.set_yticklabels([CRITERIA[c]["label"] for c in main], fontsize=8)
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        style_axes(ax, has_fira)
    if any(ax.get_legend_handles_labels()[0] for ax in axes):
        next(ax for ax in axes if ax.get_legend_handles_labels()[0]).legend(loc="lower right", fontsize=7)
    save(fig, path)


def plot_e4(path: Path, e4: dict, fams: dict) -> None:
    """Per family, delta risk at matched coverage against the family's maxprob at r* 0.03."""
    has_fira = setup_fonts()
    fig, axes = plt.subplots(1, max(len(fams), 1), figsize=(3.2 * max(len(fams), 1), 3.8), squeeze=False)
    for ax, (fam, (base, scores)) in zip(axes[0], fams.items(), strict=False):
        shown = [c for c in scores if c != base]
        xs = np.arange(len(shown))
        means = np.array([e4[c, PLOT_RISK]["d_rac"] for c in shown])
        stds = np.array([e4[c, PLOT_RISK]["d_rac_std"] for c in shown])
        ok = np.isfinite(means)
        ax.bar(xs[ok], means[ok], yerr=stds[ok], color=[CRITERIA[c]["color"] for c, k in zip(shown, ok, strict=True) if k], alpha=0.85,
               error_kw={"lw": 1, "capsize": 2})
        ax.axhline(0, color="black", lw=1)
        if not ok.any():
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center")
        ax.set_title(fam)
        ax.set_xticks(xs)
        ax.set_xticklabels([c.removeprefix(f"{fam}_") for c in shown], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"Delta risk vs {fam}_maxprob (r* = {PLOT_RISK})")
        ax.grid(alpha=0.3, axis="y")
        style_axes(ax, has_fira)
    save(fig, path)


def main() -> None:
    """Evaluate E3 and E4, write ``table.md``, ``e3.csv``, ``e4.csv``, ``e3.png`` and ``e4.png``."""
    args = parse_args()
    data = load_dataset(args.runs, args.seeds, "cifar10")
    if data is None:
        msg = f"no cifar10 dumps for seeds {args.seeds} in {args.runs}/seed*/shift"
        raise SystemExit(msg)
    active = [c for c in CRITERIA if data["units"].get(c)]
    ref, ref_note = args.reference, ""
    if not data["units"].get(ref):
        print(f"reference {ref} has no dumps for every seed; falling back to {FALLBACK_REFERENCE}")
        ref, ref_note = FALLBACK_REFERENCE, f" (fallback, `{args.reference}` is missing)"
    if ref not in active:
        msg = f"reference {ref} has no dumps"
        raise SystemExit(msg)
    labels = data["labels"]
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(len(labels)) for _ in range(args.n_splits)]
    n_ref = len(data["units"][ref])
    ctx = reference_context(data["units"][ref], labels, splits, args.delta)
    res = {c: evaluate_criterion(data["units"][c], n_ref, labels, splits, ctx, args.delta) for c in active}
    e3 = e3_rows(res, active, ref)
    fams = family_members(active)
    e4 = e4_rows(res, fams)
    args.out.mkdir(parents=True, exist_ok=True)
    md = build_markdown(args, ref, ref_note, ctx, active, e3, fams, e4)
    (args.out / "table.md").write_text(md)
    write_csvs(args.out, ref, active, e3, e4)
    plot_e3(args.out / "e3.png", e3, active, ref)
    plot_e4(args.out / "e4.png", e4, fams)
    print(md)


if __name__ == "__main__":
    main()
