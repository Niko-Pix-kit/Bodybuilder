"""Canvas preparation and source-pixel preservation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

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
    return max(multiple, int(round(value / multiple)) * multiple)


def _aspect_ratio(aspect: CompletionAspect, source_ratio: float) -> float:
    return {
        CompletionAspect.AUTO: source_ratio,
        CompletionAspect.PORTRAIT: 4 / 5,
        CompletionAspect.SQUARE: 1.0,
        CompletionAspect.LANDSCAPE: 3 / 2,
    }[aspect]


def _target_dimensions(ratio: float, long_edge: int) -> tuple[int, int]:
    if ratio >= 1:
        width = long_edge
        height = long_edge / ratio
    else:
        height = long_edge
        width = long_edge * ratio
    return _round_to_multiple(width), _round_to_multiple(height)


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

    margin_factor = 1.0 + margin_percent / 100.0
    source_ratio = source.width / source.height
    requested_ratio = _aspect_ratio(aspect, source_ratio)

    desired_width = source.width * margin_factor
    desired_height = source.height * margin_factor
    if desired_width / desired_height < requested_ratio:
        desired_width = desired_height * requested_ratio
    else:
        desired_height = desired_width / requested_ratio

    canvas_ratio = desired_width / desired_height
    canvas_width, canvas_height = _target_dimensions(canvas_ratio, target_long_edge)
    fit_scale = min(canvas_width / source.width, canvas_height / source.height)
    if margin_percent > 0:
        fit_scale /= margin_factor
    fitted_width = max(8, _round_to_multiple(source.width * fit_scale))
    fitted_height = max(8, _round_to_multiple(source.height * fit_scale))
    fitted_width = min(fitted_width, canvas_width)
    fitted_height = min(fitted_height, canvas_height)

    resized_source = source.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)
    resized_observed = observed_mask.resize((fitted_width, fitted_height), Image.Resampling.NEAREST)
    x = (canvas_width - fitted_width) // 2
    y = (canvas_height - fitted_height) // 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), (127, 127, 127))
    canvas.paste(resized_source, (x, y))
    full_observed = Image.new("L", canvas.size, 0)
    full_observed.paste(resized_observed, (x, y))
    generated = Image.fromarray(255 - np.asarray(full_observed, dtype=np.uint8), mode="L")
    placement = CanvasPlacement(
        x=x,
        y=y,
        source_width=fitted_width,
        source_height=fitted_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        original_width=source.width,
        original_height=source.height,
    )
    return PreparedCanvas(canvas, full_observed, generated, placement)


def prepare_variant_canvas(*, frame: VariantFrame, target_long_edge: int) -> PreparedCanvas:
    ratio = {
        VariantFrame.FULL_BODY: 2 / 3,
        VariantFrame.PORTRAIT: 4 / 5,
        VariantFrame.SQUARE: 1.0,
        VariantFrame.LANDSCAPE: 3 / 2,
    }[frame]
    width, height = _target_dimensions(ratio, target_long_edge)
    image = Image.new("RGB", (width, height), (127, 127, 127))
    observed = Image.new("L", image.size, 0)
    generated = Image.new("L", image.size, 255)
    placement = CanvasPlacement(0, 0, 0, 0, width, height, 0, 0)
    return PreparedCanvas(image, observed, generated, placement)


def preserve_observed_pixels(
    generated: Image.Image,
    source_canvas: Image.Image,
    generated_mask: Image.Image,
) -> Image.Image:
    generated = generated.convert("RGB")
    source_canvas = source_canvas.convert("RGB")
    generated_mask = generated_mask.convert("L")
    if generated.size != source_canvas.size or generated.size != generated_mask.size:
        raise ValueError("Generated image, source canvas, and mask must have identical dimensions")
    observed_mask = Image.fromarray(255 - np.asarray(generated_mask, dtype=np.uint8), mode="L")
    result = generated.copy()
    result.paste(source_canvas, (0, 0), observed_mask)
    return result


def generated_fraction(mask: Image.Image) -> float:
    values = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return float(values.mean())
