"""Deterministic source-image quality measurements."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class QualityScores:
    blur: float
    exposure: float
    detail: float
    resolution: float
    overall: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def measure_quality(image: Image.Image) -> QualityScores:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = _clamp(np.log1p(variance) / np.log1p(1200.0))

    luminance = gray.astype(np.float32) / 255.0
    mean = float(luminance.mean())
    clipping = float(((luminance < 0.015) | (luminance > 0.985)).mean())
    centered = 1.0 - min(1.0, abs(mean - 0.5) / 0.5)
    exposure = _clamp(centered * (1.0 - clipping) ** 0.5)

    edges = cv2.Canny(gray, 80, 180)
    edge_fraction = float((edges > 0).mean())
    detail = _clamp(edge_fraction / 0.14)

    megapixels = (image.width * image.height) / 1_000_000.0
    resolution = _clamp(np.log1p(megapixels) / np.log1p(8.0))

    components = np.maximum([blur, exposure, detail, resolution], 1e-6)
    overall = float(np.prod(components) ** 0.25)
    return QualityScores(blur, exposure, detail, resolution, _clamp(overall))
