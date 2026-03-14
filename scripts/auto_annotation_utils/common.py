"""
Shared helpers for the AgriMM auto-annotation pipeline.

These helpers intentionally keep the stage modules lightweight by centralising
common filesystem, JSONL, and dataset-structure operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for *path* if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory. Missing files return an empty list."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def load_jsonl_index(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    """Index a JSONL file by one top-level key."""
    index: dict[str, dict[str, Any]] = {}
    for record in load_jsonl_records(path):
        key = record.get(key_field)
        if key:
            index[str(key)] = record
    return index


def collect_images(image_root: Path, recursive: bool = True) -> list[Path]:
    """Collect supported image files under *image_root*."""
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    pattern = "**/*" if recursive else "*"
    images = [
        path.resolve()
        for path in sorted(image_root.glob(pattern))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return images


def infer_class_name(image_root: Path, image_path: Path) -> str | None:
    """
    Infer the class name from the first directory below *image_root*.

    Example:
      image_root=/data/images
      image_path=/data/images/maize/img1.jpg
      returns "maize"
    """
    relative_parts = image_path.resolve().relative_to(image_root.resolve()).parts
    if len(relative_parts) < 2:
        return None
    return relative_parts[0]


def discover_class_names(image_root: Path) -> list[str]:
    """Return immediate child directories under *image_root* as class names."""
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    class_names = sorted(
        child.name
        for child in image_root.iterdir()
        if child.is_dir()
    )
    return class_names


def load_class_names(class_names_file: Path | None, image_root: Path) -> list[str]:
    """
    Load class names from a file or infer them from the dataset folder layout.

    The optional file format is one class name per line.
    """
    if class_names_file is None:
        return discover_class_names(image_root)

    if not class_names_file.exists():
        raise FileNotFoundError(f"Class names file does not exist: {class_names_file}")

    class_names = [
        line.strip()
        for line in class_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(set(class_names))
