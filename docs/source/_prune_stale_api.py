"""Prune generated API pages whose documented object no longer exists.

Incremental docs builds keep ``docs/source/api`` between runs so Sphinx can
skip pages whose docstrings did not change. autosummary regenerates pages for
changed objects but never deletes pages for removed or renamed ones; such an
orphaned page makes autodoc fail in strict (-W) builds. Run this script
before ``sphinx-build``. Deleting a still-valid page is harmless: autosummary
simply regenerates it during the build.
"""

from __future__ import annotations

import importlib
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "api"


def resolves(qualified_name: str) -> bool:
    """Check whether a fully qualified name still refers to an importable object.

    Args:
        qualified_name: Dotted name of a module or of an attribute reachable
            from a module, e.g. ``probly.calibrator`` or
            ``probly.conformal_scores.lac.torch.compute_lac_score_torch``.

    Returns:
        True if the name resolves to a module or module attribute.
    """
    parts = qualified_name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


def main() -> None:
    """Delete pages in ``docs/source/api`` whose target cannot be resolved."""
    if not API_DIR.is_dir():
        return
    pruned = 0
    for page in sorted(API_DIR.glob("*.rst")):
        if not resolves(page.stem):
            page.unlink()
            print(f"Pruned stale API page: {page.name}")  # noqa: T201
            pruned += 1
    print(f"Pruned {pruned} stale API page(s).")  # noqa: T201


if __name__ == "__main__":
    main()
