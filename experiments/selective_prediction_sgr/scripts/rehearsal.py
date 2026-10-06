"""One resumable command for the whole rehearsal: start it every day, it stops at the time budget and resumes next time.

Example (PowerShell, run from the experiment directory)::

    uv run python scripts/rehearsal.py --hours 9.5 --shutdown

Jobs run in priority order and each one is skipped when its output exists, so a rerun continues where the last run
stopped (training resumes per epoch from ``runs/seed{S}/last_{stage}.pt``, dumps skip finished ``.npz`` files):

0. prerequisites (already done on a machine that ran ``run_all.py``): train ``base`` and ``dropout`` and run
   ``dump_shift.py`` for every seed;
1. train ``sngp_scratch`` seed 0, dump it, then the SNGP gate (``--sngp auto``): SNGP is only trained for the other
   seeds if its smallest Clopper-Pearson bound on the clean test set gets below the target risk (verdict in
   ``runs/rehearsal_sngp_gate.json``);
2. train ``dropout_scratch`` for every seed, then ``sngp_scratch`` seeds 1-4 (if kept);
3. dump both methods for all seeds and datasets (``dump_methods.py``);
4. ``evaluate_shift.py``, ``evaluate_options.py``, ``evaluate_calibration.py`` on the existing dumps;
5. with ``--extra-bases``: train base seeds 5-9 (only trained, not dumped).

``--hours`` sets a budget: training stops cleanly after the epoch that ends past it (exit code 75 of ``train.py``), no
dump or evaluation job is started with too little time left. ``--shutdown`` powers the PC down (Windows, 2 min delay,
abort with ``shutdown /a``) however the run ends, also after a crash. Progress is appended to ``runs/rehearsal_log.txt``.

``--arch resnet18`` switches to the one-seed architecture check (VGG-16 vs ResNet-18 for SNGP and DDU), with its own
default ``--runs runs_resnet18`` and ``--out results_resnet18`` and seed 0 only: train ``base``, ``dropout``, ``dump_shift``,
train ``ddu`` and ``sngp_scratch``, dump both, then ``check_sgr_path.py`` (output in ``runs_resnet18/check_sgr_path.txt``)::

    uv run python scripts/rehearsal.py --arch resnet18 --hours 9.5 --shutdown

``--smoke`` runs the whole queue in minutes (1 epoch on a subset, few MC samples, CIFAR-10 only); use it with its own
``--runs`` and ``--out``::

    uv run python scripts/rehearsal.py --smoke --seeds 0 1 --runs /tmp/sgr-smoke/runs --out /tmp/sgr-smoke/results
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from sgr_experiment.model import ARCHS
from sgr_experiment.utils import EXPERIMENT_DIR, run_dir

SCRIPTS = Path(__file__).resolve().parent
EXIT_DEADLINE = 75  # train.py: stopped for the time budget
MIN_DUMP_SECONDS = 20 * 60  # do not start a dump job with less time left
MIN_EVAL_SECONDS = 10 * 60  # same for an evaluation job
GATE_RISK = 0.01
GATE_DELTA = 0.001
GATE_CRITERIA = ("maxprob", "ds")
SMOKE_SUBSET = 512
SMOKE_SAMPLES = 3
CHECK_CRITERIA = ("sr_base", "sr_dropout", "mc_maxprob", "ddu_maxprob", "ddu_density", "sngp_scratch_maxprob", "sngp_scratch_ds")


@dataclass
class Job:
    """One step of the queue.

    Attributes:
        name: Display name.
        kind: ``train``, ``dump``, ``eval`` or ``gate`` (decides the minimum time left to start it).
        done: Returns True if the output exists.
        run: Executes the job and returns an exit code.
        enabled: Returns False if the job is dropped (SNGP rejected); evaluated just before the job runs.
        note: Extra text for the dry-run listing.
    """

    name: str
    kind: str
    done: Callable[[], bool]
    run: Callable[[], int]
    enabled: Callable[[], bool] = lambda: True
    note: str = ""
    status: str = field(default="todo", init=False)


class Queue:
    """Builds the job list from the command line arguments and runs it."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store the arguments, derive the deadline and build the jobs."""
        self.args = args
        self.t_start = time.time()
        self.deadline = self.t_start + args.hours * 3600 if args.hours is not None else None
        self.log_path = args.runs / "rehearsal_log.txt"
        self.gate_path = args.runs / "rehearsal_sngp_gate.json"
        self.train_extra = ["--epochs", 1, "--subset", SMOKE_SUBSET] if args.smoke else []
        self.dump_extra = ["--subset", SMOKE_SUBSET] if args.smoke else []
        if args.smoke:
            args.num_samples = SMOKE_SAMPLES
            args.datasets = ["cifar10"]
        self.jobs = self.build_jobs()

    # ---- logging ----
    def log(self, msg: str) -> None:
        """Print a timestamped line and append it to the log file."""
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if not self.args.dry_run:
            self.args.runs.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def remaining(self) -> float:
        """Seconds until the deadline (infinity without a budget)."""
        return float("inf") if self.deadline is None else self.deadline - time.time()

    # ---- gate ----
    def gate_verdict(self) -> bool | None:
        """True/False once decided (``--sngp yes/no`` decide up front), None while the gate has not run."""
        if self.args.sngp != "auto":
            return self.args.sngp == "yes"
        if self.gate_path.exists():
            return bool(json.loads(self.gate_path.read_text(encoding="utf-8"))["passed"])
        return None

    def run_gate(self) -> int:
        """Smallest prefix bound of SNGP seed 0 on the clean test set, same recipe as ``check_sgr_path.py``."""
        import numpy as np  # noqa: PLC0415

        from check_sgr_path import min_prefix_bound  # noqa: PLC0415
        from sgr_experiment.uncertainty import one_minus_max  # noqa: PLC0415

        path = run_dir(self.args.runs, 0) / "shift" / "sngp_scratch" / "cifar10.npz"
        d = np.load(path)
        labels, probs = d["labels"], d["mean_probs"]
        errors = (probs.argmax(1) != labels).astype(np.float64)
        half = len(labels) // 2  # SGR selects on half of the test set; same search level as check_sgr_path
        level = GATE_DELTA / (half - 1).bit_length()
        bounds = {}
        for k in GATE_CRITERIA:
            crit = one_minus_max(probs) if k == "maxprob" else d[k].astype(np.float64)
            bounds[k] = min_prefix_bound(crit, errors, level)
        best = min(bounds.values())
        verdict = {
            "passed": bool(best < GATE_RISK),
            "min_bound": best,
            "bounds": bounds,
            "risk": GATE_RISK,
            "delta": GATE_DELTA,
            "level": level,
            "n": len(labels),
            "test_acc": float(1.0 - errors.mean()),
        }
        self.gate_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        self.log(f"SNGP gate: {'PASSED' if verdict['passed'] else 'FAILED'}: {json.dumps(verdict)}")
        return 0

    # ---- job builders ----
    def script_cmd(self, script: str, *args: object) -> list[str]:
        """Command line of a sibling script with the current interpreter."""
        return [sys.executable, str(SCRIPTS / script), *map(str, args)]

    def exec_cmd(self, cmd: list[str]) -> int:
        """Run a command, streaming its output; returns the exit code."""
        self.log("+ " + " ".join(cmd))
        return subprocess.run(cmd, check=False).returncode

    def train_job(self, stage: str, seed: int, enabled: Callable[[], bool] = lambda: True, note: str = "") -> Job:
        """Training job of one stage and seed, done once ``{stage}.pt`` exists."""
        a = self.args
        extra = ["--deadline", self.deadline] if self.deadline is not None else []
        cmd = self.script_cmd("train.py", "--stage", stage, "--seed", seed, "--out", a.runs, "--data-dir", a.data_dir, "--arch", a.arch, *extra, *self.train_extra)
        return Job(f"train {stage} seed {seed}", "train", lambda: (run_dir(a.runs, seed) / f"{stage}.pt").exists(), lambda: self.exec_cmd(cmd), enabled, note)

    def dump_job(self, method: str | None, seeds: list[int], enabled: Callable[[], bool] = lambda: True) -> Job:
        """``dump_methods.py`` of one method (``dump_shift.py`` for None); done once every dataset file of every seed exists."""
        a = self.args
        sub = ("shift", method) if method else ("shift",)

        def done() -> bool:
            return all(run_dir(a.runs, s).joinpath(*sub, f"{d}.npz").exists() for s in seeds for d in a.datasets)

        common = ["--seeds", *seeds, "--runs", a.runs, "--data-dir", a.data_dir, "--arch", a.arch, "--num-samples", a.num_samples, "--datasets", *a.datasets, *self.dump_extra]
        cmd = self.script_cmd("dump_methods.py", "--methods", method, *common) if method else self.script_cmd("dump_shift.py", *common)
        return Job(f"dump {method or 'base/dropout (dump_shift)'} seeds {','.join(map(str, seeds))}", "dump", done, lambda: self.exec_cmd(cmd), enabled)

    def eval_job(self, script: str, out_sub: str) -> Job:
        """Evaluation on the existing dumps; done if its stamp is newer than every dump of the new methods."""
        a = self.args
        stamp = a.runs / f"rehearsal_eval_{out_sub}.stamp"

        def newest_dump() -> float:
            files = [f for m in ("dropout_scratch", "sngp_scratch") for f in a.runs.glob(f"seed*/shift/{m}/*.npz")]
            return max((f.stat().st_mtime for f in files), default=0.0)

        def done() -> bool:
            return stamp.exists() and stamp.stat().st_mtime >= newest_dump()

        def run() -> int:
            code = self.exec_cmd(self.script_cmd(script, "--runs", a.runs, "--out", a.out / out_sub, "--n-splits", a.n_splits))
            if code == 0:
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.touch()
            return code

        return Job(f"evaluate {out_sub}", "eval", done, run)

    def check_job(self) -> Job:
        """``check_sgr_path.py`` on the clean dumps (stdout also written to ``check_sgr_path.txt``); done if that file is newer than every dump."""
        a = self.args
        out_path = a.runs / "check_sgr_path.txt"

        def done() -> bool:
            dumps = [f.stat().st_mtime for f in a.runs.glob("seed*/shift/**/*.npz")]
            return out_path.exists() and out_path.stat().st_mtime >= max(dumps, default=0.0)

        def run() -> int:
            cmd = self.script_cmd("check_sgr_path.py", "--runs", a.runs, "--seeds", *a.seeds, "--criteria", *CHECK_CRITERIA)
            self.log("+ " + " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            print(res.stdout, end="", flush=True)
            print(res.stderr, end="", file=sys.stderr, flush=True)
            if res.returncode == 0:
                out_path.write_text(res.stdout, encoding="utf-8")
            return res.returncode

        return Job("check_sgr_path", "eval", done, run)

    def build_arch_check_jobs(self) -> list[Job]:
        """Queue of the architecture check: base, dropout, ddu and sngp_scratch for the seeds, then ``check_sgr_path``."""
        seeds = self.args.seeds
        jobs = [self.train_job(stage, s) for s in seeds for stage in ("base", "dropout")]
        jobs.append(self.dump_job(None, seeds))
        jobs += [self.train_job(stage, s) for s in seeds for stage in ("ddu", "sngp_scratch")]
        jobs += [self.dump_job(m, seeds) for m in ("ddu", "sngp_scratch")]
        return [*jobs, self.check_job()]

    def build_jobs(self) -> list[Job]:
        """The queue in priority order."""
        a = self.args
        if a.arch != "vgg16":
            return self.build_arch_check_jobs()
        seeds = a.seeds
        sngp_on = lambda: a.sngp != "no"  # noqa: E731
        sngp_kept = lambda: self.gate_verdict() is not False and a.sngp != "no"  # noqa: E731
        sngp_rest = [s for s in seeds if s != 0]
        jobs = [self.train_job(stage, s) for s in seeds for stage in ("base", "dropout")]
        jobs.append(self.dump_job(None, seeds))
        jobs += [self.train_job("sngp_scratch", 0, sngp_on), self.dump_job("sngp_scratch", [0], sngp_on)]
        if a.sngp == "auto":
            gate_path = run_dir(a.runs, 0) / "shift" / "sngp_scratch" / "cifar10.npz"
            jobs.append(
                Job("SNGP gate (seed 0)", "gate", self.gate_path.exists, self.run_gate, note=f"needs {gate_path.name}; verdict in {self.gate_path.name}")
            )
        jobs += [self.train_job("dropout_scratch", s) for s in seeds]
        jobs += [self.train_job("sngp_scratch", s, sngp_kept, "only if the SNGP gate passes") for s in sngp_rest]
        jobs.append(self.dump_job("dropout_scratch", seeds))
        if sngp_rest:
            jobs.append(self.dump_job("sngp_scratch", seeds, sngp_kept))
        jobs += [self.eval_job("evaluate_shift.py", "shift"), self.eval_job("evaluate_options.py", "options"), self.eval_job("evaluate_calibration.py", "calibration")]
        if a.extra_bases:
            jobs += [self.train_job("base", s, note="trained only, not dumped") for s in range(5, 10)]
        return jobs

    # ---- execution ----
    def listing(self) -> None:
        """Print every job with its done/todo/skipped state (dry run)."""
        print(f"gate verdict: {self.gate_verdict()}")
        for i, j in enumerate(self.jobs, 1):
            state = "done" if j.done() else ("skip" if not j.enabled() else "todo")
            print(f"{i:2d}. [{state}] {j.name}" + (f"  ({j.note})" if j.note else ""))

    def min_left(self, job: Job) -> float:
        """Seconds that must remain to start the job."""
        return {"dump": MIN_DUMP_SECONDS, "eval": MIN_EVAL_SECONDS}.get(job.kind, 0.0)

    def execute(self) -> str:
        """Run the jobs in order; returns why the queue ended (finished, budget, failure)."""
        for j in self.jobs:
            if j.done():
                j.status = "done"
                continue
            if not j.enabled():
                j.status = "skipped"
                self.log(f"skip {j.name} (not wanted)")
                continue
            if self.remaining() <= 0:
                return "time budget used up"
            if self.remaining() < self.min_left(j):
                return f"not enough time left to start '{j.name}' ({self.remaining() / 60:.0f} min left)"
            t0 = time.time()
            self.log(f"START {j.name}")
            code = j.run()
            dt = time.time() - t0
            if code == EXIT_DEADLINE and j.kind == "train":
                self.log(f"STOP  {j.name} after {dt / 60:.1f} min: time budget reached, progress is checkpointed")
                return "time budget used up"
            if code != 0:
                self.log(f"FAIL  {j.name} after {dt / 60:.1f} min with exit code {code}")
                return f"job '{j.name}' failed (exit code {code})"
            j.status = "done"
            self.log(f"END   {j.name} in {dt / 60:.1f} min")
        return "all jobs finished"

    def summary(self, reason: str) -> None:
        """Print done and remaining jobs and the resume command."""
        done = [j.name for j in self.jobs if j.done()]
        todo = [j.name for j in self.jobs if not j.done() and j.status != "skipped" and j.enabled()]
        self.log(f"Queue ended: {reason} after {(time.time() - self.t_start) / 3600:.2f} h.")
        self.log(f"Done ({len(done)}): " + "; ".join(done))
        self.log(f"Remaining ({len(todo)}): " + ("; ".join(todo) if todo else "none"))
        if todo:
            argv = [x for x in sys.argv[1:] if x != "--dry-run"]
            self.log("Resume with: uv run python scripts/rehearsal.py " + " ".join(argv))
        if self.args.extra_bases:
            self.log("Note: extra base seeds are trained only, not dumped.")


def shutdown_machine(reason: str) -> None:
    """Power the PC down in 2 minutes (Windows); elsewhere only print a note."""
    print(f"Shutdown requested ({reason}).", flush=True)
    if sys.platform == "win32":
        subprocess.run(["shutdown", "/s", "/t", "120"], check=False)
        print("The PC shuts down in 2 minutes; abort with `shutdown /a`.", flush=True)
    else:
        print("Not on Windows: not shutting down.", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=ARCHS, default="vgg16", help="vgg16: the full rehearsal; resnet18: one-seed architecture check.")
    p.add_argument("--hours", type=float, default=None, help="Time budget of this run in hours (default: none).")
    p.add_argument("--shutdown", action="store_true", help="Shut the PC down when the run ends for any reason (Windows).")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: 0-4 (vgg16), 0 (resnet18).")
    p.add_argument("--extra-bases", action="store_true", help="Also train base seeds 5-9 at the end (lowest priority).")
    p.add_argument("--sngp", choices=["auto", "yes", "no"], default="auto", help="auto: train SNGP for seeds > 0 only if the gate passes.")
    p.add_argument("--runs", type=Path, default=None, help="Default: runs (runs_resnet18 for resnet18).")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--out", type=Path, default=None, help="Default: results (results_resnet18 for resnet18).")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--datasets", nargs="+", default=None, help="Datasets to dump (default: all of dump_shift.py).")
    p.add_argument("--smoke", action="store_true", help="Quick end-to-end test: 1 epoch, subset, few samples, CIFAR-10 only.")
    p.add_argument("--dry-run", action="store_true", help="Print the job list with done/todo status and exit.")
    return p.parse_args()


def main() -> None:
    """Run the queue, then optionally shut down."""
    args = parse_args()
    suffix = "" if args.arch == "vgg16" else f"_{args.arch}"
    args.seeds = args.seeds or ([0, 1, 2, 3, 4] if args.arch == "vgg16" else [0])
    args.runs = args.runs or EXPERIMENT_DIR / f"runs{suffix}"
    args.out = args.out or EXPERIMENT_DIR / f"results{suffix}"
    if args.datasets is None:
        from dump_shift import ALL_DATASETS  # noqa: PLC0415

        args.datasets = list(ALL_DATASETS)
    queue = Queue(args)
    if args.dry_run:
        queue.listing()
        return
    reason, interrupted = "unknown", False
    try:
        reason = queue.execute()
    except KeyboardInterrupt:
        reason, interrupted = "interrupted by the user", True
    except Exception as exc:  # noqa: BLE001 - the user is asleep: report, then still shut down
        reason = f"crash: {exc!r}"
    try:
        queue.summary(reason)
    finally:
        sys.stdout.flush()
        if args.shutdown and not interrupted:
            shutdown_machine(reason)


if __name__ == "__main__":
    main()
