from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from canteen_checkout.config import PROJECT_ROOT


def resolve_yolo_model_reference(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parent != Path("."):
        return str((PROJECT_ROOT / path).resolve())
    return path.name


def resolve_yolo_cache(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@contextmanager
def yolo_cache_working_directory(path: Path):
    cache_dir = resolve_yolo_cache(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(cache_dir)
    try:
        yield cache_dir
    finally:
        os.chdir(previous)
