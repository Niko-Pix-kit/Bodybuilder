"""Advisory framing checks and bounded, fixed-resolution outpainting recovery.

A detector box near an edge is a reason for review, not proof of cropped anatomy.
The recovery canvas contains a generated draft, never new photographic evidence.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from PIL import Image

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class FramingAssessment:
    status: str
    message: str | None
    boxes: tuple[Box, ...] = ()
    edges: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def rank(self) -> tuple[int, int]:
        """Candidate preference only, not a confidence or visual-quality score."""
        order = {"passed": 0, "edge_contact": 1, "not_localized": 2, "multiple_subjects": 3}
        return order[self.status], len(self.edges)

    def report(self) -> dict:
        return {**asdict(self), "method": "detector_bounding_boxes", "geometry_verified": False}


def _area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(first: Box, second: Box) -> float:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    return intersection / max(1, _area(first) + _area(second) - intersection)


def assess_framing(boxes: Iterable[Box], size: tuple[int, int]) -> FramingAssessment:
    width, height = size
    if min(size) <= 0:
        raise ValueError("Framing requires positive image dimensions")
    valid = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            continue
        left, top, right, bottom = box
        clipped = (max(0, round(left)), max(0, round(top)),
                   min(width, round(right)), min(height, round(bottom)))
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            valid.append(clipped)
    distinct: list[Box] = []
    for box in sorted(valid, key=_area, reverse=True):
        # Near-identical detector proposals do not constitute a second subject.
        if not any(_iou(box, other) >= 0.85 for other in distinct):
            distinct.append(box)
    if not distinct:
        return FramingAssessment("not_localized", "The generated subject could not be localized")
    if len(distinct) > 1 and _area(distinct[1]) > _area(distinct[0]) * 0.35:
        return FramingAssessment("multiple_subjects", "More than one substantial subject may be present", tuple(distinct))
    left, top, right, bottom = distinct[0]
    mx, my = max(2, width * 0.015), max(2, height * 0.015)
    edges = tuple(name for name, touches in (
        ("left", left <= mx), ("top", top <= my),
        ("right", right >= width - mx), ("bottom", bottom >= height - my),
    ) if touches)
    if edges:
        return FramingAssessment("edge_contact", "The generated subject touches an image edge; it may be cropped", tuple(distinct), edges)
    return FramingAssessment("passed", None, tuple(distinct))


@dataclass(frozen=True)
class RecoveryCanvas:
    image: Image.Image
    missing: Image.Image
    draft_box: Box


def make_recovery_canvas(draft: Image.Image, *, scale: float = 0.70) -> RecoveryCanvas:
    """Make room for actual AI completion, not a cosmetic border-only success.

    The central draft is resampled without cropping. White mask pixels must be
    generated; black pixels retain the resampled draft. Output size stays fixed
    so this correction does not increase the diffusion memory budget.
    """
    if not math.isfinite(scale) or not 0.25 <= scale <= 0.9:
        raise ValueError("Recovery scale must be between 0.25 and 0.9")
    if min(draft.size) < 16:
        raise ValueError("Recovery requires an image of at least 16 pixels per side")
    width, height = draft.size
    fitted = draft.convert("RGB").resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    x, y = (width - fitted.width) // 2, (height - fitted.height) // 2
    box = (x, y, x + fitted.width, y + fitted.height)
    canvas = Image.new("RGB", draft.size, (127, 127, 127))
    canvas.paste(fitted, (x, y))
    missing = Image.new("L", draft.size, 255)
    missing.paste(0, box)
    return RecoveryCanvas(canvas, missing, box)
