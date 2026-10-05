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

import torch
from torch import nn

from sgr_experiment.data import make_loader
from sgr_experiment.loaders import load_base
from sgr_experiment.model import build_plain_vgg, to_mc_dropout
from sgr_experiment.utils import EXPERIMENT_DIR, default_workers, get_device, run_dir, seed_everything


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
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--subset", type=int, default=None, help="Train on a random subset of this many instances.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Deterministic eval-mode accuracy."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # noqa: PLW2901
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / total


def main() -> None:
    """Train, checkpointing every epoch and resuming from ``last.pt`` if present."""
    args = parse_args()
    defaults = {"base": (250, 0.1, 25), "dropout": (50, 0.01, 10)}[args.stage]
    args.epochs = defaults[0] if args.epochs is None else args.epochs
    args.lr = defaults[1] if args.lr is None else args.lr
    args.step_size = defaults[2] if args.step_size is None else args.step_size
    device = get_device()
    use_amp = device.type == "cuda"
    pin = device.type == "cuda"
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

    train_loader = make_loader(
        args.data_dir, train=True, batch_size=args.batch_size, workers=args.workers, pin_memory=pin,
        subset=args.subset, seed=args.seed,
    )
    test_loader = make_loader(
        args.data_dir, train=False, batch_size=500, workers=min(2, args.workers), pin_memory=pin,
    )
    loss_fn = nn.CrossEntropyLoss()
    acc = evaluate(model, test_loader, device) if start_epoch >= args.epochs else float("nan")
    new_log = not log_path.exists() or start_epoch == 0
    with log_path.open("w" if new_log else "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "train_loss", "test_acc", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            # Reseed per epoch (shuffling order and dropout masks). Persistent workers keep their own augmentation RNG
            # state, so a resumed run is statistically equivalent but not bit-identical to an uninterrupted one.
            torch.manual_seed(args.seed * 100003 + epoch)
            model.train()
            total_loss = 0.0
            n = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # noqa: PLW2901
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.item() * y.numel()
                n += y.numel()
            lr = opt.param_groups[0]["lr"]
            sched.step()
            acc = evaluate(model, test_loader, device)
            secs = time.time() - t0
            train_loss = total_loss / n
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
