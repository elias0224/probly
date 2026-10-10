"""Dump the predictions of pretrained VGG-16 and ResNet-50 on the ImageNet validation set (Sec. 5.3 of the paper).

Per model, ``{out}/{model}.npz`` holds ``labels``, the top-5 predicted classes ``top5`` (descending probability), the
softmax ``probs`` (float16), and the criteria ``sr`` (1 - max softmax) and ``top5_mass`` (1 - top-5 softmax mass) in
float64. For VGG-16, ``--mc-samples`` adds MC dropout through the two dropout layers of its classifier head (the paper's
"dropout in the last fully connected layer"; the convolutional features are computed once per image and only the head
is sampled, with independent masks per image): ``mc_top5`` of the MC mean, ``mc_maxprob`` (1 - max mean probability)
and ``mc_variance`` (variance of the predicted-class probability, the paper's MC-dropout criterion). A json file
next to it records the weights, the accuracies and the settings. Existing dumps are skipped.

``--fake N`` replaces ImageNet by N random images, for smoke tests without the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from sgr_experiment.imagenet import MODELS, PUBLISHED_ACC, imagenet_val, load_model, torch_topk_complement
from sgr_experiment.uncertainty import predicted_class_variance
from sgr_experiment.utils import EXPERIMENT_DIR, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data" / "imagenet", help="ImageNet root.")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "runs" / "imagenet")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--mc-samples", type=int, default=100, help="MC dropout samples for VGG-16; 0 skips MC dropout.")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="Seed of the dropout masks.")
    p.add_argument("--subset", type=int, default=None, help="Random subset of the validation set (smoke tests).")
    p.add_argument("--fake", type=int, default=None, help="Use N random images instead of ImageNet (smoke tests).")
    p.add_argument("--force", action="store_true", help="Overwrite existing dumps.")
    return p.parse_args()


def vgg_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Input of the classifier head of a torchvision VGG (after the conv stack, the pooling and the flattening)."""
    return torch.flatten(model.avgpool(model.features(x)), 1)


def mc_head(head: nn.Module, feats: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Softmax of ``num_samples`` passes of ``head`` with its dropout layers active, ``(samples, n, classes)``."""
    dropouts = [m for m in head.modules() if isinstance(m, nn.Dropout)]
    if not dropouts:
        msg = "the head has no dropout layer"
        raise ValueError(msg)
    for m in dropouts:
        m.train()
    try:
        return torch.stack([torch.softmax(head(feats).float(), dim=-1) for _ in range(num_samples)])
    finally:
        for m in dropouts:
            m.eval()


def dataset(args: argparse.Namespace, transform: object) -> torch.utils.data.Dataset:
    """The validation set, a random subset of it, or fake images."""
    if args.fake:
        return datasets.FakeData(size=args.fake, image_size=(3, 300, 400), num_classes=1000, transform=transform)
    ds = imagenet_val(args.data_dir, transform)
    if args.subset:
        idx = np.sort(np.random.default_rng(0).choice(len(ds), args.subset, replace=False))
        ds = Subset(ds, idx.tolist())
    return ds


def dump(name: str, args: argparse.Namespace, device: torch.device) -> None:
    """Run one model over the validation set and write ``{name}.npz`` and ``{name}.json``."""
    model, transform = load_model(name)
    model = model.to(device)
    mc = name == "vgg16" and args.mc_samples > 0
    ds = dataset(args, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    n = len(ds)
    out: dict[str, np.ndarray] = {
        "labels": np.zeros(n, dtype=np.int16),
        "top5": np.zeros((n, 5), dtype=np.int16),
        "probs": np.zeros((n, 1000), dtype=np.float16),
        "sr": np.zeros(n),
        "top5_mass": np.zeros(n),
    }
    if mc:
        out |= {"mc_top5": np.zeros((n, 5), dtype=np.int16), "mc_maxprob": np.zeros(n), "mc_variance": np.zeros(n)}
    seed_everything(args.seed)
    start, t0 = 0, time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            sl = slice(start, start + len(y))
            x = x.to(device, non_blocking=True)
            if mc:
                feats = vgg_features(model, x)
                probs = torch.softmax(model.classifier(feats).float(), dim=-1)
                samples = mc_head(model.classifier, feats, args.mc_samples)
                mean = samples.mean(0)
                out["mc_top5"][sl] = mean.topk(5, dim=-1).indices.cpu().numpy()
                out["mc_maxprob"][sl] = torch_topk_complement(mean, 1).numpy()
                out["mc_variance"][sl] = predicted_class_variance(samples).double().cpu().numpy()
            else:
                probs = torch.softmax(model(x).float(), dim=-1)
            out["labels"][sl] = y.numpy()
            out["top5"][sl] = probs.topk(5, dim=-1).indices.cpu().numpy()
            out["probs"][sl] = probs.cpu().numpy().astype(np.float16)
            out["sr"][sl] = torch_topk_complement(probs, 1).numpy()
            out["top5_mass"][sl] = torch_topk_complement(probs, 5).numpy()
            start = sl.stop
            if start % (20 * args.batch_size) < args.batch_size or start == n:
                print(f"{name}: {start}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)

    labels = out["labels"]
    acc = {
        "top1": float((out["top5"][:, 0] == labels).mean()),
        "top5": float((out["top5"] == labels[:, None]).any(1).mean()),
    }
    if mc:
        acc["mc_top1"] = float((out["mc_top5"][:, 0] == labels).mean())
        acc["mc_top5"] = float((out["mc_top5"] == labels[:, None]).any(1).mean())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{name}.npz", **out)
    meta = {
        "model": name,
        "weights": str(MODELS[name][1]),
        "transform": repr(transform),
        "n": n,
        "subset": args.subset,
        "fake": args.fake,
        "mc_samples": args.mc_samples if mc else 0,
        "seed": args.seed,
        "accuracy": acc,
        "published_accuracy": dict(zip(("top1", "top5"), PUBLISHED_ACC[name], strict=True)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: top-1 {acc['top1']:.4f}, top-5 {acc['top5']:.4f} (published {PUBLISHED_ACC[name][0]:.4f}, {PUBLISHED_ACC[name][1]:.4f})")
    if not (args.fake or args.subset) and abs(acc["top1"] - PUBLISHED_ACC[name][0]) > 0.005:
        print(f"WARNING: {name} top-1 differs from the published value by more than 0.5 points; check the label order.")


def main() -> None:
    """Dump every requested model that has no dump yet."""
    args = parse_args()
    device = get_device()
    for name in args.models:
        if (args.out / f"{name}.npz").exists() and not args.force:
            print(f"{name}: dump exists, skipping.")
            continue
        dump(name, args, device)


if __name__ == "__main__":
    main()
