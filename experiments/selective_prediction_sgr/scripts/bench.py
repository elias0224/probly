"""Measure where training time goes: device info, train step throughput per precision and memory format, augmentation.

Prints an estimated epoch time (50k images) for every variant, so a slow setup can be narrowed down.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from sgr_experiment.data import augment
from sgr_experiment.model import build_plain_vgg
from sgr_experiment.utils import get_device

EPOCH_IMAGES = 50_000


def sync(device: torch.device) -> None:
    """Wait for queued kernels so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_train(device: torch.device, *, amp: bool, channels_last: bool, batch_size: int, steps: int) -> float:
    """Return images per second of SGD train steps on random data."""
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model = build_plain_vgg().to(device).to(memory_format=fmt).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 32, 32, device=device).contiguous(memory_format=fmt)
    y = torch.randint(0, 10, (batch_size,), device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(10):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    return steps * batch_size / (time.perf_counter() - t0)


def bench_augment(device: torch.device, batch_size: int, steps: int) -> float:
    """Return images per second of the GPU augmentation alone."""
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(batch_size, 3, 32, 32, device=device)
    for _ in range(5):
        augment(x, gen)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        augment(x, gen)
    sync(device)
    return steps * batch_size / (time.perf_counter() - t0)


def main() -> None:
    """Print device info and timings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    device = get_device()
    print(f"torch {torch.__version__}, device {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU {props.name}, compute capability {props.major}.{props.minor}, "
            f"{props.total_memory / 2**30:.1f} GiB, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}"
        )
        torch.backends.cudnn.benchmark = True
    for amp in (False, True):
        for channels_last in (False, True):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ips = bench_train(device, amp=amp, channels_last=channels_last, batch_size=a.batch_size, steps=a.steps)
            mem = f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device.type == "cuda" else ""
            print(
                f"train amp={amp!s:5} channels_last={channels_last!s:5}: {ips:8.0f} img/s"
                f" -> {EPOCH_IMAGES / ips:6.1f} s/epoch{mem}"
            )
    ips = bench_augment(device, a.batch_size, a.steps)
    print(f"augmentation only: {ips:8.0f} img/s -> {EPOCH_IMAGES / ips:6.1f} s/epoch")


if __name__ == "__main__":
    main()
