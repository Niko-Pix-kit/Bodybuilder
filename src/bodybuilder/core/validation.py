"""Detect technical image failures, not identity accuracy or aesthetic quality."""

from __future__ import annotations

import numpy as np
from PIL import Image


class InvalidGenerationError(RuntimeError):
    """The model returned non-finite, empty, mask-like, or unchanged output."""


def mask_like(image: Image.Image) -> bool:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    black = np.all(pixels <= 1, axis=-1)
    white = np.all(pixels >= 254, axis=-1)
    return float((black | white).mean()) > 0.999


def validate_generation(
    image: Image.Image,
    source_canvas: Image.Image,
    generated_mask: Image.Image,
) -> None:
    if image.size != source_canvas.size or image.size != generated_mask.size:
        raise InvalidGenerationError("The model returned the wrong image dimensions")
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    selected = np.asarray(generated_mask.convert("L")) > 127
    if not selected.any():
        return
    region = pixels[selected]
    if mask_like(image):
        raise InvalidGenerationError("The model returned black/white blocks instead of a photograph")
    if len(region) >= 64 and float(region.std(axis=0).max()) < 0.5:
        raise InvalidGenerationError("The missing region is almost uniform; reconstruction failed")
    original = np.asarray(source_canvas.convert("RGB"), dtype=np.float32)[selected]
    if float(np.abs(region - original).mean()) < 0.1:
        raise InvalidGenerationError("The model left the missing region unchanged")


def decode_model_output(output: object) -> Image.Image:
    """Check floating-point output before NaNs can silently turn into black pixels."""
    if isinstance(output, Image.Image):
        return output.convert("RGB")
    array = np.asarray(output)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise InvalidGenerationError("The model produced invalid or non-finite pixels")
    if not np.issubdtype(array.dtype, np.floating):
        raise InvalidGenerationError("Unexpected model output; expected RGB floats in [0, 1]")
    if array.min() < -0.01 or array.max() > 1.01:
        raise InvalidGenerationError("The model returned pixels outside the expected range")
    return Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8))
