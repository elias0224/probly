"""Other ways to choose the accept/abstain rule, evaluated on the saved prediction dumps of ``evaluate_shift.py``.

Same protocol as ``evaluate_shift.py``: ``n_splits`` permutations of the clean test set, the first half selects (or
calibrates) the rule, the second half is the test half; shifted sets use the same test-half indices and the OOD
subsets are drawn with the same generator. Every mode is a function returning an acceptance mask for any dataset:
the threshold modes accept ``criterion <= threshold``, the probability modes (Chow, conformal) accept from that
dataset's own probabilities. Nothing needs a GPU, only the ``.npz`` dumps are read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
import time
import warnings

from evaluate import setup_fonts
from evaluate_shift import (
    CRITERIA,
    EPS,
    ID_RISKS,
    SHIFT_DATASETS,
    SHIFT_RISKS,
    Tables,
    active_criteria,
    load_dataset,
    mean_of,
    save,
    sgr_selector_threshold,
    style_axes,
)
from fixed_threshold import fit_temperature, scale
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.conformal_scores import APSScore, lac_score  # noqa: E402
from probly.selective_prediction import CoverageSelector, ThresholdSelector  # noqa: E402
from sgr_experiment.metrics import apply_threshold, threshold_for_risk  # noqa: E402
from sgr_experiment.options import (  # noqa: E402
    aps_singleton_from_top_two,
    conformal_qhat,
    lac_singleton_from_top_two,
    ltt_bonferroni_threshold,
    ltt_fixed_sequence_threshold,
    top_two,
)
from sgr_experiment.shift import CORRUPTIONS, OOD_DATASETS, SEVERITIES, corrupted_name  # noqa: E402
from sgr_experiment.uncertainty import one_minus_max  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

MODE_INFO = {
    "emp": ("emp", "largest selection-half coverage with empirical risk <= r* (no guarantee, reference)", "#9e9e9e"),
    "sgr": ("sgr", "SGR, Algorithm 1 of Geifman and El-Yaniv (2017) via probly's SGRSelector", "#1e88e5"),
    "cov": ("cov", "label-free CoverageSelector at the selection-half coverage of emp (reference)", "#bdbdbd"),
    "ltt_bonf": ("LTT Bonf", "Learn-then-Test (Angelopoulos et al., 2021), binomial p-values, Bonferroni over G candidates", "#e53935"),
    "ltt_fs": ("LTT FS", "Learn-then-Test with multi-start fixed-sequence testing, K starts walking up in coverage", "#fb8c00"),
    "chow_raw": ("Chow raw", "Chow's rule (1970): ThresholdSelector(c = r*) on 1 - max prob, no fitting, no guarantee", "#43a047"),
    "chow_ts": ("Chow TS", "Chow's rule after temperature scaling fitted on the selection half", "#00897b"),
    "conf_lac": ("Conf LAC", "split conformal with the LAC score at alpha = r* (Sadinle et al., 2019), accept singleton sets", "#8e24aa"),
    "conf_aps": ("Conf APS", "split conformal with the non-randomized APS score at alpha = r*, accept singleton sets", "#5e35b1"),
}
ALL_MODES = list(MODE_INFO)
THRESHOLD_MODES = {"emp", "sgr", "cov", "ltt_bonf", "ltt_fs"}
PROB_MODES = {"chow_raw", "chow_ts", "conf_lac", "conf_aps"}
# Criteria that have a probability source: 1 - max prob of sr_*, mc_maxprob, ens_maxprob, {method}_maxprob, swa_maxprob.
PROB_CRITERIA = {c for c in CRITERIA if c.endswith("_maxprob") or c in ("sr_base", "sr_dropout")}
PROB_METHODS = ["finetune", "swag", "laplace", "ddu", "vbll", "sngp"]
PLOT_RISKS = [0.01, 0.03]
LOG_FLOOR = 1e-30


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "options")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of SGR and LTT.")
    p.add_argument("--grid-size", type=int, default=100, help="Number G of LTT candidate thresholds.")
    p.add_argument("--starts", type=int, default=10, help="Number K of fixed-sequence starts.")
    p.add_argument("--criteria", nargs="+", default=None, help="Criteria to evaluate (default: all active ones).")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES, help="Modes to evaluate.")
    return p.parse_args()


def load_probs(runs: Path, seeds: list[int], name: str) -> dict[str, list[np.ndarray]] | None:
    """Probabilities behind each criterion of ``PROB_CRITERIA``, one array ``(n, classes)`` per unit.

    The units are aligned with ``load_dataset``: one per seed, except ``ens_maxprob`` (one unit). Returns None if a
    seed has no file for the dataset.
    """
    files = [runs / f"seed{s}" / "shift" / f"{name}.npz" for s in seeds]
    if not all(f.exists() for f in files):
        return None
    ds = [np.load(f) for f in files]
    out: dict[str, list[np.ndarray]] = {c: [] for c in ("sr_base", "sr_dropout", "mc_maxprob")}
    for d in ds:
        out["sr_base"].append(d["softmax_base"])
        out["sr_dropout"].append(d["softmax_dropout"])
        out["mc_maxprob"].append(d["mean_probs"])
    for m in PROB_METHODS:
        mfiles = [runs / f"seed{sd}" / "shift" / m / f"{name}.npz" for sd in seeds]
        if not all(f.exists() for f in mfiles):
            continue
        for f in mfiles:
            md = np.load(f)
            out.setdefault(f"{m}_maxprob", []).append(md["mean_probs"])
            if m == "swag":
                out.setdefault("swa_maxprob", []).append(md["softmax_swa"])
    out["ens_maxprob"] = [np.stack([d["softmax_base"] for d in ds]).astype(np.float64).mean(0)]
    return {k: v for k, v in out.items() if k in PROB_CRITERIA}


def verify_predictions(data: dict[str, dict], probs: dict[str, dict]) -> int:
    """Check that the argmax of every probability unit equals the prediction of ``load_dataset``; returns the count."""
    n = 0
    for name, entry in probs.items():
        for c, plist in entry.items():
            units = data[name]["units"][c]
            if len(units) != len(plist):
                msg = f"{name} {c}: {len(plist)} probability units but {len(units)} criterion units."
                raise SystemExit(msg)
            for (_, pred), p in zip(units, plist, strict=True):
                if not np.array_equal(p.argmax(1), pred):
                    msg = f"{name} {c}: argmax of the probabilities differs from the prediction of load_dataset."
                    raise SystemExit(msg)
                n += 1
    return n


class ProbContext:
    """Probability based rules (Chow, conformal) for one criterion, unit and split.

    Calibration (temperature, conformal scores) uses the selection half of the clean set only. Derived arrays are
    memoized per ``(dataset, index array)`` so that the three shift risks and all modes share them.
    """

    def __init__(self, sources: Callable[[str], np.ndarray], labels: np.ndarray, sel: np.ndarray, modes: set[str]) -> None:
        self.sources = sources
        self.cache: dict[tuple, object] = {}
        p_sel = np.asarray(sources("cifar10")[sel], dtype=np.float64)
        y_sel = labels[sel]
        self.temperature = fit_temperature(np.log(np.maximum(p_sel, LOG_FLOOR)), y_sel) if "chow_ts" in modes else 1.0
        self.lac_cal = np.asarray(lac_score(p_sel, y_sel)) if "conf_lac" in modes else np.empty(0)
        self.aps_cal = np.asarray(APSScore(randomized=False)(p_sel, y_sel)) if "conf_aps" in modes else np.empty(0)

    def omm(self, name: str, idx: np.ndarray, *, ts: bool) -> np.ndarray:
        """``1 - max prob`` on ``idx`` of a dataset, raw or after temperature scaling."""
        key = (name, id(idx), ts)
        if key not in self.cache:
            p = np.asarray(self.sources(name)[idx], dtype=np.float64)
            if ts:
                p = scale(np.log(np.maximum(p, LOG_FLOOR)), self.temperature)
            self.cache[key] = one_minus_max(p)
        return self.cache[key]  # ty: ignore[invalid-return-type]

    def top2(self, name: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two largest probabilities on ``idx`` of a dataset."""
        key = (name, id(idx), "top2")
        if key not in self.cache:
            self.cache[key] = top_two(self.sources(name)[idx])
        return self.cache[key]  # ty: ignore[invalid-return-type]

    def rule(self, mode: str, r: float) -> Callable[[str, np.ndarray], np.ndarray]:
        """Acceptance function ``(dataset, index array) -> mask`` of a probability mode at level ``r``."""
        if mode == "chow_raw":
            sel_ = ThresholdSelector(r)
            return lambda name, idx: np.asarray(sel_.select(self.omm(name, idx, ts=False)))
        if mode == "chow_ts":
            sel_ = ThresholdSelector(r)
            return lambda name, idx: np.asarray(sel_.select(self.omm(name, idx, ts=True)))
        if mode == "conf_lac":
            q = conformal_qhat(self.lac_cal, r)
            return lambda name, idx: lac_singleton_from_top_two(*self.top2(name, idx), q)
        q = conformal_qhat(self.aps_cal, r)
        return lambda name, idx: aps_singleton_from_top_two(*self.top2(name, idx), q)


def _stats(mask: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    """Selective risk (NaN if nothing is accepted) and coverage of an acceptance mask."""
    return (float(loss[mask].mean()) if mask.any() else float("nan")), float(mask.mean())


def run_protocol(
    data: dict[str, dict], probs: dict[str, dict], crits: list[str], modes: list[str], args: argparse.Namespace
) -> tuple[dict[tuple, list[float]], float]:
    """Evaluate every criterion, unit, split, r* and mode; returns the values per result key and the seconds used."""
    t0 = time.perf_counter()
    acc: dict[tuple, list[float]] = defaultdict(list)
    clean = data["cifar10"]
    labels = clean["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    ood = [d for d in OOD_DATASETS if d in data]
    shifted = [d for d in SHIFT_DATASETS if d in data]
    rng_ood = np.random.default_rng(args.split_seed + 1)
    subsets = {d: [rng_ood.choice(len(data[d]["labels"]), n - half, replace=len(data[d]["labels"]) < n - half) for _ in splits] for d in ood}
    mode_set = set(modes)
    need_prob = bool(mode_set & PROB_MODES)

    for c in crits:
        ood_c = [d for d in ood if data[d]["units"].get(c)]
        shifted_c = [d for d in shifted if data[d]["units"].get(c)]
        has_probs = need_prob and c in PROB_CRITERIA and bool(probs["cifar10"].get(c))
        for u, (crit, pred) in enumerate(clean["units"][c]):
            loss = (pred != labels).astype(np.float64)
            shift_loss = {d: (data[d]["units"][c][u][1] != labels).astype(np.float64) for d in shifted_c}
            crit_of = {d: data[d]["units"][c][u][0] for d in [*shifted_c, *ood_c]}
            crit_of["cifar10"] = crit
            for si, perm in enumerate(splits):
                sel, test = perm[:half], perm[half:]
                ctx = None
                if has_probs:
                    ctx = ProbContext(lambda name, c=c, u=u: probs[name][c][u], labels, sel, mode_set)
                for r in sorted(set(ID_RISKS) | set(SHIFT_RISKS)):
                    rules: dict[str, Callable[[str, np.ndarray], np.ndarray]] = {}
                    thresholds: dict[str, float] = {}
                    thresholds["emp"] = threshold_for_risk(crit[sel], loss[sel], r)
                    if "sgr" in mode_set:
                        thresholds["sgr"] = sgr_selector_threshold(crit[sel], loss[sel], r, args.delta)[0]
                    if "cov" in mode_set:
                        target = apply_threshold(crit[sel], loss[sel], thresholds["emp"])[1]
                        thresholds["cov"] = CoverageSelector(target).calibrate(crit[sel]).threshold if target > 0 else -np.inf
                    if "ltt_bonf" in mode_set:
                        thresholds["ltt_bonf"] = ltt_bonferroni_threshold(crit[sel], loss[sel], r, args.delta, args.grid_size)
                    if "ltt_fs" in mode_set:
                        thresholds["ltt_fs"] = ltt_fixed_sequence_threshold(crit[sel], loss[sel], r, args.delta, args.grid_size, args.starts)
                    for mode in modes:
                        if mode in THRESHOLD_MODES:
                            rules[mode] = lambda name, idx, thr=thresholds[mode]: crit_of[name][idx] <= thr
                        elif ctx is not None:
                            rules[mode] = ctx.rule(mode, r)
                    for mode, rule in rules.items():
                        if r in ID_RISKS:
                            risk, cov = _stats(rule("cifar10", test), loss[test])
                            acc["id", mode, r, c, "risk"].append(risk)
                            acc["id", mode, r, c, "cov"].append(cov)
                            acc["id", mode, r, c, "viol"].append(float(risk > r + EPS) if np.isfinite(risk) else 0.0)
                        if r in SHIFT_RISKS:
                            for d in ood_c:
                                acc["ood_acc", d, mode, r, c].append(float(rule(d, subsets[d][si]).mean()))
                            for d in shifted_c:
                                risk, cov = _stats(rule(d, test), shift_loss[d][test])
                                acc["shift", d, mode, r, c, "risk"].append(risk)
                                acc["shift", d, mode, r, c, "cov"].append(cov)
    return acc, time.perf_counter() - t0


def header_text(modes: list[str], n_seeds: int, args: argparse.Namespace) -> list[str]:
    """Markdown header explaining the protocol and every mode."""
    lines = [
        "# Choosing the accept/abstain rule (CIFAR-10 VGG-16)",
        "",
        f"{n_seeds} seeds x {args.n_splits} random 5k/5k splits of the clean test set; entries are mean +- std over all seed x",
        "split pairs. Every rule is fitted on the selection half and evaluated on the test half; shifted sets use the same",
        "test-half images, and the OOD share is measured on a random subset of as many OOD images as the test half. Ensemble",
        "criteria (`ens_*`) use the base models as members, so their spread comes from the random splits only. `viol` is the",
        f"share of seed x split pairs with test-half risk > r*. LTT uses delta = {args.delta}, G = {args.grid_size} candidates and",
        f"K = {args.starts} fixed-sequence starts. `n/a` marks modes that need probabilities for criteria without any (entropies, density, ...).",
        "",
        "Modes:",
        "",
    ]
    for m in modes:
        lines.append(f"- `{m}`: {MODE_INFO[m][1]}.")
    lines += [
        "",
        "References: Geifman and El-Yaniv (2017), Selective classification for deep neural networks (SGR); Angelopoulos, Bates,",
        "Candes, Jordan and Lei (2021), Learn then Test (LTT); Chow (1970), On optimum recognition error and reject tradeoff;",
        "Sadinle, Lei and Wasserman (2019), Least ambiguous set-valued classifiers with bounded error levels (LAC).",
        "LTT certifies the candidates of a label-free grid, so with probability >= 1 - delta the true risk of the returned rule is",
        "<= r*. Chow's rule has no finite-sample guarantee: it is only as good as the calibration of the probabilities. Conformal",
        "sets satisfy P(y not in set) <= alpha, hence P(error and accepted) <= alpha and the selective risk is <= alpha / coverage,",
        "a weaker guarantee than SGR and LTT. APS is the non-randomized version of probly's `aps_score`.",
        "",
    ]
    return lines


def build_tables(acc: dict, data: dict[str, dict], crits: list[str], modes: list[str], out: Path, n_seeds: int, args: argparse.Namespace) -> str:
    """Write all tables as csv files and ``table.md``; returns the markdown."""
    t = Tables(out)
    ood = [d for d in OOD_DATASETS if d in data]
    for r in ID_RISKS:
        specs = []
        for m in modes:
            lab = MODE_INFO[m][0]
            specs += [{"header": f"{lab} risk", "digits": 4, "mode": "ms"}, {"header": f"{lab} cov", "digits": 4, "mode": "ms"}, {"header": f"{lab} viol", "digits": 0, "mode": "pct"}]
        rows = [([c], [("id", m, r, c, k) for m in modes for k in ("risk", "cov", "viol")]) for c in crits]
        t.add(f"id_r{r}", f"a) ID test half at r* = {r}: risk, coverage and violation share (risk > r*)", ["criterion"], rows, specs, acc)
    for d in ood:
        specs = [{"header": f"{MODE_INFO[m][0]} r*={r}", "digits": 3, "mode": "ms"} for r in SHIFT_RISKS for m in modes]
        rows = [([c], [("ood_acc", d, m, r, c) for r in SHIFT_RISKS for m in modes]) for c in crits]
        t.add(f"ood_{d}", f"b) OOD {d}: share of OOD images accepted at the clean rule", ["criterion"], rows, specs, acc)
    for corruption in CORRUPTIONS:
        sets = [(s, corrupted_name(corruption, s)) for s in SEVERITIES if corrupted_name(corruption, s) in data]
        if not sets:
            continue
        for r in SHIFT_RISKS:
            specs = []
            for m in modes:
                specs += [{"header": f"{MODE_INFO[m][0]} risk", "digits": 4, "mode": "ms", "mark": r}, {"header": f"{MODE_INFO[m][0]} cov", "digits": 3, "mode": "mean"}]
            rows = [([str(s), c], [("shift", d, m, r, c, k) for m in modes for k in ("risk", "cov")]) for s, d in sets for c in crits]
            t.add(f"shift_{corruption}_r{r}", f"c) Shift: {corruption}, r* = {r} (* marks mean risk > r*)", ["severity", "criterion"], rows, specs, acc)
    md = "\n".join(header_text(modes, n_seeds, args) + t.md)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def plot_coverage(path: Path, acc: dict, crits: list[str], modes: list[str], flag: float) -> None:
    """ID test coverage per criterion with grouped bars per mode; hatched bars have a violation share > ``flag``."""
    fig, axes = plt.subplots(len(PLOT_RISKS), 1, figsize=(max(9.0, 0.55 * len(crits) * len(modes) / 3), 4.2 * len(PLOT_RISKS)), sharex=True)
    width = 0.8 / len(modes)
    for ax, r in zip(np.atleast_1d(axes), PLOT_RISKS, strict=True):
        for j, m in enumerate(modes):
            vals = [np.asarray(acc.get(("id", m, r, c, "cov"), [np.nan]), dtype=float) for c in crits]
            viol = [mean_of(acc, ("id", m, r, c, "viol")) for c in crits]
            xs = np.arange(len(crits)) + (j - (len(modes) - 1) / 2) * width
            for x, v, vi in zip(xs, vals, viol, strict=True):
                if not np.isfinite(v).any():
                    continue
                ax.bar(
                    x, np.nanmean(v), width, yerr=np.nanstd(v), color=MODE_INFO[m][2], edgecolor="white" if vi <= flag else "black",
                    hatch="////" if vi > flag else None, lw=0.5, error_kw={"lw": 0.7},
                )
        ax.set_ylabel(f"test coverage, r* = {r}")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.6, len(crits) - 0.4)
        ax.grid(alpha=0.25, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MODE_INFO[m][2]) for m in modes]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="////"))
    labels = [MODE_INFO[m][0] for m in modes] + [f"violation share > {flag:g}"]
    np.atleast_1d(axes)[0].legend(handles, labels, frameon=False, ncol=min(5, len(labels)), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    np.atleast_1d(axes)[-1].set_xticks(np.arange(len(crits)), crits, rotation=40, ha="right")
    for ax in np.atleast_1d(axes):
        style_axes(ax)
    save(fig, path)


def print_summary(acc: dict, crits: list[str], modes: list[str]) -> None:
    """Print coverage and violation share per criterion and mode at r* = 0.01 and 0.03."""
    for r in PLOT_RISKS:
        print(f"\nr* = {r}: ID test coverage / violation % per mode")
        print(f"{'criterion':16s}" + "".join(f" {MODE_INFO[m][0]:>14s}" for m in modes))
        for c in crits:
            cells = []
            for m in modes:
                cov = mean_of(acc, ("id", m, r, c, "cov"))
                cells.append("n/a" if np.isnan(cov) else f"{cov:.3f}/{mean_of(acc, ('id', m, r, c, 'viol')) * 100:.0f}%")
            print(f"{c:16s}" + "".join(f" {x:>14s}" for x in cells))


def main() -> None:
    """Load the dumps, run all modes, write tables, plot and summary."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(f.parent.parent.name[4:]) for f in args.runs.glob("seed*/shift/cifar10.npz"))
    if not seeds:
        msg = f"No seed with shift/cifar10.npz under {args.runs}; run scripts/dump_shift.py first."
        raise SystemExit(msg)
    data: dict[str, dict] = {}
    probs: dict[str, dict] = {}
    for name in ["cifar10", *OOD_DATASETS, *SHIFT_DATASETS]:
        loaded = load_dataset(args.runs, seeds, name)
        if loaded is None:
            print(f"skipping {name}: not dumped for all seeds {seeds}.")
            continue
        data[name] = loaded
        probs[name] = load_probs(args.runs, seeds, name) or {}
    n_checked = verify_predictions(data, probs)
    print(f"Verified the argmax of {n_checked} probability units against the predictions of load_dataset.")
    available = active_criteria(data)
    crits = available if args.criteria is None else [c for c in args.criteria if c in available]
    unknown = sorted(set(args.criteria or []) - set(available))
    if unknown:
        print(f"WARNING: criteria not available and skipped: {unknown}")
    if not crits:
        raise SystemExit("No criterion left to evaluate.")
    modes = [m for m in ALL_MODES if m in args.modes]
    setup_fonts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        acc, seconds = run_protocol(data, probs, crits, modes, args)
    build_tables(acc, data, crits, modes, args.out, len(seeds), args)
    flag = 10 * args.delta
    plot_coverage(args.out / "coverage_by_mode.png", acc, crits, modes, flag)
    print_summary(acc, crits, modes)
    n_units = sum(len(data["cifar10"]["units"][c]) for c in crits)
    print(f"\nProtocol took {seconds:.1f}s for {len(crits)} criteria, {n_units} units x {args.n_splits} splits, {len(data)} datasets.")
    print(f"Wrote table.md, coverage_by_mode.png and the csv tables to {args.out}")


if __name__ == "__main__":
    main()
