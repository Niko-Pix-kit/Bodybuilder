"""Image discovery, explicit missing-pixel masks, and atomic persistence."""

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
except ImportError:
    register_heif_opener = None
if register_heif_opener is not None:
    register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}


def scan_images(folder: Path, *, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        if path.name.lower().endswith((".mask.png", "_observed_mask.png", "_generated_mask.png")):
            continue
        # Walk only inside the selected source tree, excluding earlier BodyBuilder runs.
        if any((parent / "run_manifest.json").exists() for parent in path.parents if parent == folder or folder in parent.parents):
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: str(path).casefold())


def fragment_mask_path(path: Path) -> Path:
    return path.with_name(path.name + ".mask.png")


def load_fragment(path: Path) -> tuple[Image.Image, Image.Image]:
    """White in the observed mask is real evidence. Never guess from black/white RGB."""
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        alpha = oriented.getchannel("A")
        observed = alpha.point(lambda value: 255 if value == 255 else 0)
        rgb = Image.new("RGB", oriented.size, (127, 127, 127))
        rgb.paste(oriented.convert("RGB"), mask=alpha)
    sidecar = fragment_mask_path(path)
    if sidecar.exists():
        with Image.open(sidecar) as mask_source:
            missing = ImageOps.exif_transpose(mask_source).convert("L")
            if missing.size != rgb.size:
                raise ValueError(f"Missing-area mask has different dimensions: {sidecar.name}")
            observed.paste(0, mask=missing.point(lambda value: 255 if value > 127 else 0))
    if observed.getbbox() is None:
        raise ValueError(f"No observed pixels remain in {path.name}")
    # Replace known missing areas by a neutral placeholder, not invented evidence.
    rgb.paste((127, 127, 127), mask=ImageOps.invert(observed))
    return rgb, observed


def load_rgb(path: Path) -> Image.Image:
    """Decode a preview without treating transparency as recoverable evidence."""
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", oriented.size, "white")
        return Image.alpha_composite(background, oriented).convert("RGB")


def safe_stem(value: str, *, fallback: str = "image") -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    _atomic_replace(path, lambda temporary: image.save(temporary, format="PNG", pnginfo=info))


def write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_replace(path, lambda temporary: temporary.write_text(encoded, encoding="utf-8"))
