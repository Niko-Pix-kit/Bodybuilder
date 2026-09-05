"""Shared data contracts; reference photographs remain separate images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    path: Path
    width: int
    height: int
    file_size: int
    blur_score: float
    exposure_score: float
    detail_score: float
    resolution_score: float
    quality_score: float
    faces: tuple[FaceBox, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class GenerationRequest:
    canvas: Image.Image
    generated_mask: Image.Image
    reference_board: Image.Image
    face_reference_board: Image.Image | None
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance_scale: float
    strength: float
    reference_fidelity: float
    width: int
    height: int
    fully_synthetic: bool = False
    reference_images: tuple[Image.Image, ...] = ()


@dataclass(slots=True)
class StitchResult:
    image: Image.Image
    observed_mask: Image.Image
    used_paths: list[Path] = field(default_factory=list)
    rejected_paths: list[Path] = field(default_factory=list)
