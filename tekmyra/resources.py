"""Portable lookup for data shipped with :mod:`tekmyra`.

Editable installs read from the repository. Wheel installs read from the
platform data directory populated by setuptools. ``TEKMYRA_DATA_DIR``
may point at an unpacked data root for controlled deployments.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path, PurePosixPath, PureWindowsPath

_DATA_ENV = "TEKMYRA_DATA_DIR"
_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_INSTALLED_ROOT = Path(sysconfig.get_path("data")) / "share" / "tekmyra"


def data_roots() -> tuple[Path, ...]:
    """Return candidate roots in explicit override, source, install order."""
    roots: list[Path] = []
    configured = os.environ.get(_DATA_ENV)
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend((_SOURCE_ROOT, _INSTALLED_ROOT))

    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def resource_path(*parts: str) -> Path:
    """Resolve a packaged resource and fail with searched locations."""
    def unsafe(part: str) -> bool:
        if not part:
            return True
        portable_paths = (PurePosixPath(part), PureWindowsPath(part))
        return any(path.anchor or ".." in path.parts for path in portable_paths)

    if not parts or any(unsafe(part) for part in parts):
        raise ValueError("resource parts must be non-empty relative paths")

    candidates = tuple(root.joinpath(*parts) for root in data_roots())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"tekmyra resource {'/'.join(parts)!r} was not found; "
        f"searched: {searched}"
    )


def resource_dir(*parts: str) -> Path:
    """Resolve a shipped resource directory."""
    path = resource_path(*parts)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path
