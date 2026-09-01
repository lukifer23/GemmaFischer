from __future__ import annotations

from pathlib import Path


def bundled_path(relative: str) -> Path:
    """Resolve a checked-out resource or its wheel-packaged equivalent."""
    repository_path = Path(__file__).parents[2] / relative
    if repository_path.is_file():
        return repository_path
    packaged_path = Path(__file__).with_name("resources") / relative
    if packaged_path.is_file():
        return packaged_path
    return repository_path
