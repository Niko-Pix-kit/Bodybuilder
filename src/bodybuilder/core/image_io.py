"""Robust image discovery, decoding, and atomic persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, PngImagePlugin

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    # HEIF remains unavailable, but every other format continues to work.
    pass

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
    ".heic",
    ".heif",
}


def scan_images(folder: Path, *, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    ]
    return sorted(paths, key=lambda path: str(path).casefold())


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode == "RGBA":
            background = Image.new("RGB", oriented.size, "white")
            background.paste(oriented, mask=oriented.getchannel("A"))
            return background
        if oriented.mode == "LA":
            background = Image.new("RGB", oriented.size, "white")
            background.paste(oriented.convert("L"), mask=oriented.getchannel("A"))
            return background
        return oriented.convert("RGB")


def safe_stem(value: str, *, fallback: str = "image") -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    clean = clean.strip("._-")
    return clean[:100] or fallback


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 100_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique path near {path}")


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_png(image: Image.Image, path: Path, metadata: dict[str, Any] | None = None) -> None:
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("BodyBuilder", json.dumps(metadata, ensure_ascii=False, sort_keys=True))

    def writer(temporary: Path) -> None:
        image.save(temporary, format="PNG", pnginfo=info, optimize=True)

    _atomic_replace(path, writer)


def write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def writer(temporary: Path) -> None:
        temporary.write_text(encoded, encoding="utf-8")

    _atomic_replace(path, writer)
