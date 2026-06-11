"""Normalize tracked-file mtimes from content checksums for incremental builds.

A fresh CI checkout stamps every file with the current time, which would make
Sphinx treat all sources and autodoc dependencies as outdated even though the
restored build cache is current. Run this script after restoring the docs
build cache and before ``sphinx-build``: it compares a checksum of every
git-tracked file against the manifest stored next to the cached build output.
Files whose content is unchanged get a fixed old mtime so Sphinx keeps their
cached output; changed or new files keep their fresh checkout mtime and are
rebuilt. The manifest is then rewritten so the cache saved by this run
describes the sources it was built from. Without a manifest (cold build) all
files keep fresh mtimes and a full build runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "build" / "file-checksums.json"

# Fixed timestamp for unchanged files: 2000-01-01 UTC. Any value older than
# the read times recorded in the restored Sphinx environment works.
OLD_MTIME = 946684800


def main() -> None:
    """Reset mtimes of files whose checksum matches the cached manifest."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    tracked = [name for name in listing.stdout.decode().split("\0") if name]
    previous: dict[str, str] = {}
    if MANIFEST.is_file():
        previous = json.loads(MANIFEST.read_text())
    current: dict[str, str] = {}
    unchanged = 0
    for name in tracked:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        current[name] = digest
        if previous.get(name) == digest:
            os.utime(path, (OLD_MTIME, OLD_MTIME))
            unchanged += 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(current, sort_keys=True, indent=0) + "\n")
    fresh = len(current) - unchanged
    print(f"Reset mtimes of {unchanged} unchanged file(s); {fresh} kept fresh.")  # noqa: T201


if __name__ == "__main__":
    main()
