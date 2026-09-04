"""Visual provenance outputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _labelled(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (size[0], size[1] - 32), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    panel.paste(fitted, ((size[0] - fitted.width) // 2, 32 + (size[1] - 32 - fitted.height) // 2))
    ImageDraw.Draw(panel).text((10, 10), label, fill="black", font=ImageFont.load_default())
    return panel


def create_comparison(
    source_canvas: Image.Image,
    generated_mask: Image.Image,
    result: Image.Image,
) -> Image.Image:
    width = 420
    height = max(300, round(width * result.height / max(1, result.width)) + 32)
    source = _labelled(source_canvas, "Observed canvas", (width, height))
    mask_rgb = Image.merge("RGB", (generated_mask, generated_mask, generated_mask))
    mask = _labelled(mask_rgb, "Generated mask (white = synthetic)", (width, height))
    output = _labelled(result, "Result", (width, height))
    sheet = Image.new("RGB", (width * 3, height), (230, 230, 230))
    sheet.paste(source, (0, 0))
    sheet.paste(mask, (width, 0))
    sheet.paste(output, (width * 2, 0))
    return sheet


def save_jpeg(image: Image.Image, path: Path, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.convert("RGB").save(temporary, format="JPEG", quality=quality, optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
