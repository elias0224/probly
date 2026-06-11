"""Sphinx helper utilities for the probly documentation build."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT_HELPERS = Path(__file__).resolve().parents[2]

# Local packages whose source changes can alter example outputs. All are
# importable from this repository checkout (editable installs or sys.path
# entries added in conf.py), so their module files map back to working-tree
# files that can be re-hashed later.
TRACKED_PACKAGES = ("probly", "flextype", "pytraverse", "probly_benchmark", "examples")

DEPS_SUFFIX = ".deps.json"


def code_hash(path: Path) -> str:
    """Hash the behavior-relevant code of a Python source file.

    The file is parsed to an AST and docstrings are stripped before hashing,
    so edits to docstrings, comments, or formatting do not change the hash;
    only changes to executable code do. Files that fail to parse are hashed
    byte-wise instead.

    Args:
        path: Path of the Python source file.

    Returns:
        Hex digest identifying the file's executable code.
    """
    source = path.read_bytes()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return hashlib.sha256(source).hexdigest()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _tracked_module_names() -> list[str]:
    """Return names of currently imported modules from tracked local packages."""
    return [name for name in sys.modules if name.partition(".")[0] in TRACKED_PACKAGES]


def reset_probly_dep_tracking(gallery_conf: dict[str, Any], fname: str | None, when: str) -> None:
    """Record which local modules a gallery example imports.

    Registered in ``sphinx_gallery_conf["reset_modules"]`` (as a dotted string,
    so the config value stays picklable and stable across builds) together with
    ``reset_modules_order = "both"``. Before each example, all tracked local
    packages are removed from ``sys.modules`` so the example imports them
    afresh. Afterwards, every local module the example loaded is written to a
    ``<example>.py.deps.json`` manifest next to the generated gallery output,
    mapping each dependency file to its :func:`code_hash`. Lazily registered
    backend modules are captured naturally because recording happens at
    runtime. ``_invalidate_stale_examples.py`` reads these manifests before
    the next build and forces re-execution of examples whose dependencies
    changed.

    Args:
        gallery_conf: The sphinx-gallery configuration dictionary.
        fname: Base filename of the example being run, or None when called
            between gallery directories.
        when: Either "before" or "after" example execution.
    """
    if fname is None:
        return
    if when == "before":
        for name in _tracked_module_names():
            del sys.modules[name]
        return
    deps: dict[str, str] = {}
    for name in sorted(_tracked_module_names()):
        module_file = getattr(sys.modules[name], "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        try:
            relpath = path.relative_to(REPO_ROOT_HELPERS)
        except ValueError:
            continue  # not from this working tree, cannot map to source changes
        deps[relpath.as_posix()] = code_hash(path)
    src_dir = Path(gallery_conf.get("src_dir", Path(__file__).resolve().parent))
    gallery_dirs = gallery_conf.get("gallery_dirs", "auto_examples")
    if isinstance(gallery_dirs, str):
        gallery_dirs = [gallery_dirs]
    matches = [match for gallery in gallery_dirs for match in (src_dir / gallery).rglob(fname)]
    if len(matches) != 1:
        print(  # noqa: T201
            f"dep tracking: expected exactly one gallery copy of {fname}, "
            f"found {len(matches)}; skipping dependency manifest"
        )
        return
    manifest = matches[0].with_name(matches[0].name + DEPS_SUFFIX)
    manifest.write_text(json.dumps(deps, sort_keys=True, indent=0) + "\n")


def make_linkcode_resolve(repo_root: Path) -> Callable[[str, dict[str, str]], str | None]:
    """Return a ``linkcode_resolve`` function bound to *repo_root*.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        A callable suitable for assignment to ``linkcode_resolve`` in ``conf.py``.
    """

    def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
        """Return a URL to the source for the given Python object for Sphinx linkcode.

        Args:
            domain: The domain, e.g. "py".
            info: Info dict containing keys like "module" and "fullname".

        Returns:
            The URL to the source file, or None if it cannot be resolved.
        """
        if domain != "py" or not info.get("module"):
            return None
        try:
            module = importlib.import_module(info["module"])
            obj = module
            for part in info["fullname"].split("."):
                obj = getattr(obj, part)
            fn = inspect.getsourcefile(obj)
            _src, lineno = inspect.getsourcelines(obj)
            if fn is None:
                return None
            relpath = Path(fn).relative_to(repo_root)
        except (ModuleNotFoundError, AttributeError, TypeError, OSError, ValueError):
            return None

        base = "https://github.com/n-teGruppe/probly"
        branch = "sphinx_gallery"
        return f"{base}/blob/{branch}/{relpath}#L{lineno}"

    return linkcode_resolve
