"""Force re-execution of cached gallery examples whose dependencies changed.

sphinx-gallery only re-runs an example when the example file itself changes
(MD5 check against the cached output in ``docs/source/auto_examples``);
changes to probly source code that alter example *outputs* go unnoticed. To
close that gap, the ``reset_probly_dep_tracking`` hook in ``_sphinx_helpers``
records the local modules each example imports while it runs, in a
``<example>.py.deps.json`` manifest next to the cached output. Run this
script before ``sphinx-build``: it deletes the ``.md5`` marker of every
cached example whose recorded dependencies are missing or hash differently
now (and of examples without a manifest, e.g. caches predating dependency
tracking), which makes sphinx-gallery execute those examples again.
"""

from __future__ import annotations

import json
from pathlib import Path

from _sphinx_helpers import DEPS_SUFFIX, REPO_ROOT_HELPERS, code_hash

GALLERY_DIR = Path(__file__).resolve().parent / "auto_examples"


def stale_reason(manifest: Path, hash_cache: dict[str, str | None]) -> str | None:
    """Determine whether a cached example's dependencies are out of date.

    Args:
        manifest: Path of the example's ``.deps.json`` dependency manifest.
        hash_cache: Mutable cache mapping repo-relative dependency paths to
            their current code hash (None for files that no longer exist),
            shared across manifests to hash each file only once.

    Returns:
        A human-readable reason if the example must be re-executed, else None.
    """
    if not manifest.is_file():
        return "no dependency manifest"
    deps: dict[str, str] = json.loads(manifest.read_text())
    for relpath, recorded_hash in sorted(deps.items()):
        if relpath not in hash_cache:
            dep_file = REPO_ROOT_HELPERS / relpath
            hash_cache[relpath] = code_hash(dep_file) if dep_file.is_file() else None
        current_hash = hash_cache[relpath]
        if current_hash is None:
            return f"{relpath} was removed"
        if current_hash != recorded_hash:
            return f"{relpath} changed"
    return None


def main() -> None:
    """Delete ``.md5`` markers of cached examples whose dependencies changed."""
    if not GALLERY_DIR.is_dir():
        return
    hash_cache: dict[str, str | None] = {}
    invalidated = 0
    for md5_file in sorted(GALLERY_DIR.rglob("*.py.md5")):
        example = md5_file.with_suffix("")
        reason = stale_reason(example.with_name(example.name + DEPS_SUFFIX), hash_cache)
        if reason is None:
            continue
        md5_file.unlink()
        invalidated += 1
        print(f"Will re-run {example.relative_to(GALLERY_DIR)}: {reason}")  # noqa: T201
    print(f"Invalidated {invalidated} cached example(s).")  # noqa: T201


if __name__ == "__main__":
    main()
