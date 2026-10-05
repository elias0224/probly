"""Train the VGG-16 on CIFAR-10 in two stages (each resumable).

Stage "base" trains the plain VGG; stage "dropout" applies probly's MC dropout to the trained base and fine-tunes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from sgr_experiment.data import load_cifar10, normalize, train_batches
from sgr_experiment.loaders import load_base
from sgr_experiment.model import build_plain_vgg, to_mc_dropout
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, run_dir, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=["base", "dropout"], default="base")
    p.add_argument("--epochs", type=int, default=None, help="Default: 250 (base) or 50 (dropout).")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="Default: 0.1 (base) or 0.01 (dropout).")
    p.add_argument("--step-size", type=int, default=None, help="Halve the lr every this many epochs. Default: 25 / 10.")
    p.add_argument("--p", type=float, default=0.5, help="Dropout probability of the inserted layers (stage dropout).")
    p.add_argument("--subset", type=int, default=None, help="Train on a random subset of this many instances.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, batch_size: int = 1000) -> float:
    """Deterministic eval-mode accuracy on normalized test tensors that live on the model's device."""
    model.eval()
    correct = 0
    for start in range(0, len(y_test), batch_size):
        logits = model(x_test[start : start + batch_size])
        correct += (logits.argmax(-1) == y_test[start : start + batch_size]).sum().item()
    return correct / len(y_test)


def main() -> None:
    """Train, checkpointing every epoch and resuming from ``last.pt`` if present."""
    args = parse_args()
    defaults = {"base": (250, 0.1, 25), "dropout": (50, 0.01, 10)}[args.stage]
    args.epochs = defaults[0] if args.epochs is None else args.epochs
    args.lr = defaults[1] if args.lr is None else args.lr
    args.step_size = defaults[2] if args.step_size is None else args.step_size
    device = get_device()
    use_amp = device.type == "cuda"
    out = run_dir(args.out, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"last_{args.stage}.pt"
    log_path = out / f"log_{args.stage}.csv"

    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.stage == "base":
        model = build_plain_vgg()
    else:
        base_path = out / "base.pt"
        if not base_path.exists():
            msg = f"{base_path} not found; train stage base first."
            raise SystemExit(msg)
        model = to_mc_dropout(load_base(base_path), p=args.p)
    # No channels-last: it was 4-5x slower for this network on an RTX 2070 Super (see scripts/bench.py).
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    acc = evaluate(model, x_test, y_test) if start_epoch >= args.epochs else float("nan")
    new_log = not log_path.exists() or start_epoch == 0
    with log_path.open("w" if new_log else "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "train_loss", "test_acc", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            # Reseed per epoch (shuffling, augmentation and dropout masks), so a resumed run continues like an
            # uninterrupted one up to nondeterminism of the GPU kernels.
            torch.manual_seed(args.seed * 100003 + epoch)
            gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + epoch)
            model.train()
            total_loss = torch.zeros((), device=device)
            n = 0
            for x, y in train_batches(x_train, y_train, batch_size=args.batch_size, generator=gen):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.detach().float() * y.numel()
                n += y.numel()
            lr = opt.param_groups[0]["lr"]
            sched.step()
            acc = evaluate(model, x_test, y_test)
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
                },
                tmp,
            )
            tmp.replace(ckpt_path)

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f"{args.stage}.pt")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=EXPERIMENT_DIR
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    meta = {
        "stage": args.stage,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_step_size": args.step_size,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": args.batch_size,
        "dropout_p": args.p if args.stage == "dropout" else None,
        "subset": args.subset,
        "test_acc": acc,
        "git_commit": commit,
    }
    (out / f"{args.stage}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
