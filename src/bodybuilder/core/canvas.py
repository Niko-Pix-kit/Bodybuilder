"""Prepare non-degenerate AI canvases while keeping an explicit evidence mask."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageOps

from bodybuilder.config import CompletionAspect, VariantFrame


@dataclass(frozen=True, slots=True)
class CanvasPlacement:
    x: int
    y: int
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    original_width: int
    original_height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class PreparedCanvas:
    image: Image.Image
    observed_mask: Image.Image
    generated_mask: Image.Image
    placement: CanvasPlacement


def _round_to_multiple(value: float, multiple: int = 8) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _target_dimensions(ratio: float, long_edge: int) -> tuple[int, int]:
    # A narrow forehead/limb fragment must not create an 8-pixel-wide SDXL canvas.
    ratio = max(512 / long_edge, min(long_edge / 512, ratio))
    if ratio >= 1:
        return long_edge, max(512, _round_to_multiple(long_edge / ratio))
    return max(512, _round_to_multiple(long_edge * ratio)), long_edge


def prepare_outpaint_canvas(
    source: Image.Image,
    observed_mask: Image.Image,
    *,
    aspect: CompletionAspect,
    margin_percent: int,
    target_long_edge: int,
) -> PreparedCanvas:
    source = source.convert("RGB")
    observed_mask = observed_mask.convert("L")
    if source.size != observed_mask.size:
        raise ValueError("Source image and observed mask must have identical dimensions")
    if observed_mask.getbbox() is None:
        raise ValueError("The source has no observed pixels")
    ratio = {CompletionAspect.AUTO: source.width / source.height,
             CompletionAspect.PORTRAIT: 4 / 5, CompletionAspect.SQUARE: 1.0,
             CompletionAspect.LANDSCAPE: 3 / 2}[aspect]
    width, height = _target_dimensions(ratio, target_long_edge)
    scale = min(width / source.width, height / source.height) / (1 + margin_percent / 100)
    fitted_width = max(1, min(width, round(source.width * scale)))
    fitted_height = max(1, min(height, round(source.height * scale)))
    fitted = source.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)
    mask = observed_mask.resize(fitted.size, Image.Resampling.NEAREST)
    x, y = (width - fitted_width) // 2, (height - fitted_height) // 2
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    canvas.paste(fitted, (x, y))
    observed = Image.new("L", canvas.size, 0)
    observed.paste(mask, (x, y))
    return PreparedCanvas(canvas, observed, ImageOps.invert(observed), CanvasPlacement(
        x, y, fitted_width, fitted_height, width, height, source.width, source.height))


def prepare_variant_canvas(*, frame: VariantFrame, target_long_edge: int) -> PreparedCanvas:
    ratio = {VariantFrame.FULL_BODY: 2 / 3, VariantFrame.PORTRAIT: 4 / 5,
             VariantFrame.SQUARE: 1.0, VariantFrame.LANDSCAPE: 3 / 2}[frame]
    width, height = _target_dimensions(ratio, target_long_edge)
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    return PreparedCanvas(canvas, Image.new("L", canvas.size, 0), Image.new("L", canvas.size, 255),
                          CanvasPlacement(0, 0, 0, 0, width, height, 0, 0))


def preserve_observed_pixels(generated: Image.Image, source_canvas: Image.Image,
                             generated_mask: Image.Image) -> Image.Image:
    if generated.size != source_canvas.size or generated.size != generated_mask.size:
        raise ValueError("Generated image, source canvas, and mask must have identical dimensions")
    result = generated.convert("RGB").copy()
    result.paste(source_canvas.convert("RGB"), (0, 0), ImageOps.invert(generated_mask.convert("L")))
    return result


def generated_fraction(mask: Image.Image) -> float:
    return float(np.asarray(mask.convert("L"), dtype=np.float32).mean() / 255)
