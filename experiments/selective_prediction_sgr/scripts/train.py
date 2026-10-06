"""Train the VGG-16 (or, with ``--arch resnet18``, a ResNet-18) on CIFAR-10 in stages (each resumable).

Stage "base" trains the plain VGG. All other stages start from the trained ``base.pt`` and fine-tune it: "dropout"
applies probly's MC dropout, "finetune" keeps the plain model (control for the extra training), "swag" wraps it in
probly's SWAG and collects weight snapshots, "ddu" applies DDU (spectral normalization, the density head is fitted
later), "vbll" replaces the last Linear layer by a variational Bayesian last layer, "sngp" applies SNGP (spectral
normalization and a random-feature Gaussian process last layer; the precision matrix is reset every epoch).
Two SNGP variants test whether the short fine-tune limits SNGP: "sngp_long" fine-tunes base with the 50-epoch budget
of finetune/dropout, and "sngp_scratch" trains SNGP from a fresh VGG with the base schedule (no ``base.pt`` needed).
Likewise "dropout_scratch" trains the VGG with probly's MC dropout from scratch with the base schedule.

``--deadline`` (unix timestamp) stops training cleanly after the epoch that ends past it (exit code 75, the checkpoint
``last_{stage}.pt`` is complete); rerunning the same command resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from probly.losses import vbll_loss
from probly.method.swag import collect_swag
from probly.method.sngp import reset_precision_matrix
from probly.method.vbll import find_vbll_layer
from sgr_experiment.data import load_cifar10, normalize, train_batches
from sgr_experiment.loaders import load_base
from sgr_experiment.model import ARCHS, build_plain, to_ddu, to_mc_dropout, to_sngp, to_swag, to_vbll
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, run_dir, seed_everything


# (epochs, lr, lr step size in epochs) per stage; swag keeps the lr constant.
DEFAULTS = {
    "base": (250, 0.1, 25),
    "dropout": (50, 0.01, 10),
    "finetune": (50, 0.01, 10),
    "swag": (20, 0.01, 10**6),
    "ddu": (20, 0.01, 10),
    "vbll": (20, 0.01, 10),
    "sngp": (20, 0.01, 10),
    "sngp_long": (50, 0.01, 10),
    "sngp_scratch": (250, 0.1, 25),
    "dropout_scratch": (250, 0.1, 25),
}
EXIT_DEADLINE = 75  # stopped for the time budget, resume later
SNGP_STAGES = ("sngp", "sngp_long", "sngp_scratch")
KL_WEIGHT = 1.0 / 50000  # VBLL: 1 / training set size


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=ARCHS, default="vgg16", help="Network; a runs directory holds one arch only (checked).")
    p.add_argument("--stage", choices=list(DEFAULTS), default="base")
    p.add_argument("--epochs", type=int, default=None, help="Default: 250 (base, sngp_scratch, dropout_scratch), 50 (dropout, finetune, sngp_long), 20 (others).")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="Default: 0.1 (base, sngp_scratch, dropout_scratch) or 0.01 (others).")
    p.add_argument("--step-size", type=int, default=None, help="Halve the lr every this many epochs. Default: 25 (base, sngp_scratch, dropout_scratch), constant (swag), else 10.")
    p.add_argument("--p", type=float, default=0.5, help="Dropout probability of the inserted layers (stages dropout, dropout_scratch).")
    p.add_argument("--swag-start", type=int, default=5, help="First epoch (1-based) after which SWAG collects a snapshot.")
    p.add_argument("--swag-max-rank", type=int, default=20)
    p.add_argument("--swag-scale", type=float, default=0.5)
    p.add_argument("--sn-coeff", type=float, default=3.0, help="Spectral normalization coefficient (stage ddu).")
    p.add_argument("--sngp-norm-multiplier", type=float, default=6.0, help="Spectral norm bound (stage sngp).")
    p.add_argument(
        "--sngp-init-std",
        type=float,
        default=None,
        help="Std of the SNGP random-feature weights. Default: 1.0 (sngp_scratch), 0.05 (fine-tuning a trained base).",
    )
    p.add_argument("--sngp-momentum", type=float, default=-1.0, help="Precision matrix momentum; < 0 accumulates per epoch.")
    p.add_argument("--vbll-parameterization", default="dense", choices=["diagonal", "dense", "lowrank"])
    p.add_argument("--deadline", type=float, default=None, help="Unix timestamp; stop (exit code 75) once it has passed.")
    p.add_argument("--subset", type=int, default=None, help="Train on a random subset of this many instances.")
    return p.parse_args()


def replace_with_retry(src: Path, dst: Path, attempts: int = 10, wait: float = 1.0) -> None:
    """Rename ``src`` to ``dst``, retrying while Windows reports the target as locked (antivirus, indexer, sync)."""
    for i in range(attempts):
        try:
            src.replace(dst)
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait)
        else:
            return


def check_arch(found: str, wanted: str, path: Path) -> None:
    """Exit with an error if a finished stage or checkpoint was made for another architecture (missing means vgg16)."""
    if found != wanted:
        msg = f"{path} was made for arch {found}, but --arch is {wanted}; use a separate --out directory per arch."
        raise SystemExit(msg)


def logits_of(stage: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logits of a stage's model (DDU and VBLL models do not return plain logits)."""
    if stage == "ddu":
        return model.classification_head(model.encoder(x))
    if stage == "vbll" or stage in SNGP_STAGES:
        return model(x)[0]
    return model(x)


@torch.no_grad()
def evaluate(stage: str, model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, batch_size: int = 1000) -> float:
    """Deterministic eval-mode accuracy on normalized test tensors that live on the model's device."""
    model.eval()
    correct = 0
    for start in range(0, len(y_test), batch_size):
        logits = logits_of(stage, model, x_test[start : start + batch_size])
        correct += (logits.argmax(-1) == y_test[start : start + batch_size]).sum().item()
    return correct / len(y_test)


def main() -> None:
    """Train, checkpointing every epoch and resuming from ``last.pt`` if present."""
    args = parse_args()
    defaults = DEFAULTS[args.stage]
    args.epochs = defaults[0] if args.epochs is None else args.epochs
    args.lr = defaults[1] if args.lr is None else args.lr
    args.step_size = defaults[2] if args.step_size is None else args.step_size
    if args.sngp_init_std is None:
        # 1.0 is the full RFF kernel for training from scratch; 0.05 keeps cos near linear to preserve trained features.
        args.sngp_init_std = 1.0 if args.stage == "sngp_scratch" else 0.05
    device = get_device()
    # DDU (power iteration) and VBLL (Cholesky based loss) are kept in fp32.
    use_amp = device.type == "cuda" and args.stage not in ("ddu", "vbll", *SNGP_STAGES)
    out = run_dir(args.out, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"last_{args.stage}.pt"
    log_path = out / f"log_{args.stage}.csv"

    for name in {args.stage, "base"}:  # a finished stage, and the base a fine-tune stage would start from
        meta_path = out / f"{name}.json"
        if meta_path.exists():
            check_arch(json.loads(meta_path.read_text(encoding="utf-8")).get("arch", "vgg16"), args.arch, meta_path)

    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    sngp_kwargs = {
        "norm_multiplier": args.sngp_norm_multiplier,
        "random_feature_init_std": args.sngp_init_std,
        "momentum": args.sngp_momentum,
    }
    if args.stage == "base":
        model = build_plain(args.arch)
    elif args.stage == "dropout_scratch":
        model = to_mc_dropout(build_plain(args.arch), p=args.p)
    elif args.stage == "sngp_scratch":
        model = to_sngp(build_plain(args.arch), **sngp_kwargs)
    else:
        base_path = out / "base.pt"
        if not base_path.exists():
            msg = f"{base_path} not found; train stage base first."
            raise SystemExit(msg)
        base = load_base(base_path, args.arch)
        model = {
            "dropout": lambda: to_mc_dropout(base, p=args.p),
            "finetune": lambda: base,
            "swag": lambda: to_swag(base, max_rank=args.swag_max_rank, scale=args.swag_scale),
            "ddu": lambda: to_ddu(base, sn_coeff=args.sn_coeff),
            "vbll": lambda: to_vbll(base, parameterization=args.vbll_parameterization),
            "sngp": lambda: to_sngp(base, **sngp_kwargs),
            "sngp_long": lambda: to_sngp(base, **sngp_kwargs),
        }[args.stage]()
    # No channels-last: it was 4-5x slower for this network on an RTX 2070 Super (see scripts/bench.py).
    model = model.to(device)
    vbll_features: dict[str, torch.Tensor] = {}
    if args.stage == "vbll":
        vbll_layer = find_vbll_layer(model)
        vbll_layer.register_forward_pre_hook(lambda _m, inputs: vbll_features.update(x=inputs[0]))
        # No weight decay on the variational parameters: it would shrink the posterior covariance parameters.
        rest = [p for p in model.parameters() if all(p is not q for q in vbll_layer.parameters())]
        groups = [{"params": rest}, {"params": list(vbll_layer.parameters()), "weight_decay": 0.0}]
    else:
        groups = [{"params": list(model.parameters())}]
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        check_arch(ckpt.get("arch", "vgg16"), args.arch, ckpt_path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        print(f"Resuming seed {args.seed} at epoch {start_epoch} (of {args.epochs}).")
    if start_epoch >= args.epochs and (out / f"{args.stage}.pt").exists():
        print(f"Seed {args.seed} stage {args.stage} already trained for {start_epoch} epochs, nothing to do.")
        return

    # The whole dataset lives on the device and is augmented there, so no DataLoader workers are needed.
    x_train, y_train = load_cifar10(args.data_dir, train=True, device=device)
    if args.subset is not None and args.subset < len(y_train):
        idx = torch.from_numpy(np.random.default_rng(args.seed).permutation(len(y_train))[: args.subset]).to(device)
        x_train, y_train = x_train[idx], y_train[idx]
    x_test, y_test = load_cifar10(args.data_dir, train=False, device=device)
    x_test = normalize(x_test)
    loss_fn = nn.CrossEntropyLoss()
    acc = evaluate(args.stage, model, x_test, y_test) if start_epoch >= args.epochs else float("nan")
    new_log = not log_path.exists() or start_epoch == 0
    with log_path.open("w" if new_log else "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "train_loss", "test_acc", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs):
            if args.deadline is not None and time.time() >= args.deadline:
                print(f"Deadline passed before epoch {epoch + 1}; resume later.")
                sys.exit(EXIT_DEADLINE)
            t0 = time.time()
            # Reseed per epoch (shuffling, augmentation and dropout masks), so a resumed run continues like an
            # uninterrupted one up to nondeterminism of the GPU kernels.
            torch.manual_seed(args.seed * 100003 + epoch)
            gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + epoch)
            model.train()
            if args.stage in SNGP_STAGES and args.sngp_momentum < 0:
                reset_precision_matrix(model)  # accumulate the precision over exactly one epoch
            total_loss = torch.zeros((), device=device)
            n = 0
            for x, y in train_batches(x_train, y_train, batch_size=args.batch_size, generator=gen):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    if args.stage == "vbll":
                        model(x)
                        loss = vbll_loss(vbll_layer, vbll_features["x"], y, KL_WEIGHT)
                    else:
                        loss = loss_fn(logits_of(args.stage, model, x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.detach().float() * y.numel()
                n += y.numel()
            lr = opt.param_groups[0]["lr"]
            sched.step()
            if args.stage == "swag" and (epoch + 1 >= args.swag_start or epoch + 1 == args.epochs):  # always the last
                collect_swag(model)
            acc = evaluate(args.stage, model, x_test, y_test)
            secs = time.time() - t0
            train_loss = total_loss.item() / n
            writer.writerow([epoch + 1, f"{train_loss:.5f}", f"{acc:.5f}", f"{lr:.6f}", f"{secs:.1f}"])
            fh.flush()
            print(f"seed {args.seed} epoch {epoch + 1}/{args.epochs} loss {train_loss:.4f} acc {acc:.4f} ({secs:.1f}s)")
            if not torch.isfinite(torch.tensor(train_loss)):
                msg = "Training diverged (non-finite loss); try --lr 0.05."
                raise RuntimeError(msg)
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "arch": args.arch,
                },
                tmp,
            )
            replace_with_retry(tmp, ckpt_path)
            if args.deadline is not None and time.time() >= args.deadline and epoch + 1 < args.epochs:
                print(f"Deadline passed after epoch {epoch + 1}/{args.epochs}; checkpoint saved, resume later.")
                sys.exit(EXIT_DEADLINE)

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f"{args.stage}.pt")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=EXPERIMENT_DIR
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "stage": args.stage,
        "arch": args.arch,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_step_size": args.step_size,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": args.batch_size,
        "dropout_p": args.p if args.stage in ("dropout", "dropout_scratch") else None,
        "swag": {"start": args.swag_start, "max_rank": args.swag_max_rank, "scale": args.swag_scale}
        if args.stage == "swag"
        else None,
        "sn_coeff": args.sn_coeff if args.stage == "ddu" else None,
        "sngp": {"norm_multiplier": args.sngp_norm_multiplier, "init_std": args.sngp_init_std, "momentum": args.sngp_momentum}
        if args.stage in SNGP_STAGES
        else None,
        "vbll_parameterization": args.vbll_parameterization if args.stage == "vbll" else None,
        "subset": args.subset,
        "test_acc": acc,
        "git_commit": commit,
    }
    (out / f"{args.stage}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
