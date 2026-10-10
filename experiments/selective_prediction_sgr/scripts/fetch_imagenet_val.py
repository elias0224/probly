"""Download the ILSVRC2012 validation set from Hugging Face and unpack it into ``data/imagenet/val/<wnid>/``.

The dataset ``ILSVRC/imagenet-1k`` is gated: accept its terms on huggingface.co and log in (``hf auth login``) first.
Only the validation parquet shards (about 6.7 GB) are downloaded, into ``data/imagenet/hf/``. The JPEG bytes are
written unchanged, one folder per class named by its wnid, so ``sgr_experiment.imagenet.imagenet_val`` reads them as
an image folder with the usual sorted-wnid label order. The wnids come from the dataset's ``classes.py``, which is
parsed, not executed. The result is checked: 50,000 images, 1,000 classes, 50 per class.

Needs ``huggingface_hub`` and ``pyarrow``, which the experiment does not depend on::

    uv run --with huggingface_hub --with pyarrow python scripts/fetch_imagenet_val.py
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.parquet as pq

from sgr_experiment.utils import EXPERIMENT_DIR

REPO = "ILSVRC/imagenet-1k"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=EXPERIMENT_DIR / "data" / "imagenet")
    p.add_argument("--revision", default=None, help="Dataset revision (commit or branch); default the main branch.")
    return p.parse_args()


def wnids(cache: Path, revision: str | None) -> list[str]:
    """The 1000 wnids in label order, read from the literal ``IMAGENET2012_CLASSES`` dict in ``classes.py``."""
    path = hf_hub_download(REPO, "classes.py", repo_type="dataset", revision=revision, local_dir=cache)
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "IMAGENET2012_CLASSES" for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call):  # OrderedDict({...}) or OrderedDict([...])
                value = value.args[0]
            keys = [k.value for k in value.keys] if isinstance(value, ast.Dict) else [e.elts[0].value for e in value.elts]
            if len(keys) != 1000 or keys != sorted(keys):
                msg = "classes.py does not list 1000 wnids in sorted order"
                raise ValueError(msg)
            return keys
    msg = "IMAGENET2012_CLASSES not found in classes.py"
    raise ValueError(msg)


def main() -> None:
    """Download, unpack and check."""
    args = parse_args()
    cache = args.root / "hf"
    val = args.root / "val"
    names = wnids(cache, args.revision)
    snapshot_download(REPO, repo_type="dataset", revision=args.revision, allow_patterns=["data/validation-*"], local_dir=cache)
    shards = sorted((cache / "data").glob("validation-*.parquet"))
    if not shards:
        msg = f"no validation shards in {cache / 'data'}"
        raise SystemExit(msg)
    for name in names:
        (val / name).mkdir(parents=True, exist_ok=True)
    written = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "label"])
        for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist(), strict=True):
            stem = Path(image.get("path") or f"val_{written:08d}").stem
            target = val / names[label] / f"{stem}.JPEG"
            if not target.exists():
                target.write_bytes(image["bytes"])
            written += 1
        print(f"{shard.name}: {written} images", flush=True)

    counts = {name: sum(1 for _ in (val / name).iterdir()) for name in names}
    total = sum(counts.values())
    bad = {k: v for k, v in counts.items() if v != 50}
    print(f"{total} images in {len(counts)} classes under {val}")
    if total != 50000 or bad:
        msg = f"unexpected layout: {total} images, classes without 50 images: {list(bad.items())[:5]}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
