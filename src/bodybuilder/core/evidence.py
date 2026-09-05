"""Explicit regional evidence, never face recognition or cross-person matching.

A guide belongs to an oriented source image and is invalidated if that image changes.
Only original, unmasked pixels can become reference patches. Target face geometry
is a layout hint, not recovered evidence. Eye-pair detection is deliberately only
an editable heuristic; cropped mouths and hairstyles require user-labelled boxes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from bodybuilder.core.image_io import is_generated_image, load_fragment, write_json

Box = tuple[float, float, float, float]
ROLES = ("eyes", "mouth", "hair")


@dataclass(frozen=True, slots=True)
class Region:
    role: str
    box: Box
    origin: str = "user"


@dataclass(slots=True)
class Guide:
    regions: list[Region] = field(default_factory=list)
    face_box: Box | None = None
    auto_eyes: bool = True


@dataclass(slots=True)
class DetailReference:
    role: str
    path: Path
    box: Box
    image: Image.Image
    origin: str
    score: float
    source_sha256: str

    def metadata(self) -> dict:
        return {"role": self.role, "source": str(self.path), "box": list(self.box),
                "origin": self.origin, "source_sha256": self.source_sha256,
                "selection_score": round(self.score, 4)}


def guide_path(path: Path) -> Path:
    return path.with_name(path.name + ".evidence.json")


def source_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_box(value: object, size: tuple[int, int], *, outside: bool = False) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A facial guide rectangle must contain four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError("Facial guide coordinates must be numbers")
    x0, y0, x1, y1 = box = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in box) or x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("Facial guide rectangles must be finite and at least 4 pixels wide/high")
    width, height = size
    lower, upper = (-4, 5) if outside else (0, 1)
    if not (lower * width <= x0 < x1 <= upper * width and
            lower * height <= y0 < y1 <= upper * height):
        raise ValueError("Facial guide rectangle is outside the permitted image area")
    return box


def read_guide(path: Path, size: tuple[int, int]) -> Guide:
    sidecar = guide_path(path)
    if not sidecar.exists():
        return Guide()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError(f"Unsupported facial guide: {sidecar.name}")
    if data.get("image_size") != list(size) or data.get("source_sha256") != source_digest(path):
        raise ValueError(f"Source changed since its facial guide was saved: {path.name}. Reset the guide.")
    entries = data.get("regions", [])
    if not isinstance(entries, list) or len(entries) > 24:
        raise ValueError(f"Invalid facial reference regions: {sidecar.name}")
    regions = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ROLES:
            raise ValueError(f"Unknown facial reference role: {sidecar.name}")
        regions.append(Region(entry["role"], valid_box(entry.get("box"), size)))
    face = data.get("face_box")
    return Guide(regions, valid_box(face, size, outside=True) if face is not None else None,
                 bool(data.get("auto_eyes", True)))


def save_guide(path: Path, guide: Guide) -> None:
    image, observed = load_fragment(path)
    for region in guide.regions:
        if region.role not in ROLES:
            raise ValueError("Unknown facial reference role")
        box = valid_box(region.box, image.size)
        if observed.crop(box).getextrema() != (255, 255):
            raise ValueError("Select only visible details, excluding the painted/transparent missing area")
    if guide.face_box is not None:
        valid_box(guide.face_box, image.size, outside=True)
    write_json(guide_path(path), {
        "schema": 1, "image_size": list(image.size), "source_sha256": source_digest(path),
        "auto_eyes": guide.auto_eyes, "face_box": guide.face_box,
        "regions": [{"role": r.role, "box": r.box} for r in guide.regions],
    })


def eye_pair(image: Image.Image, observed: Image.Image) -> tuple[Box, Box] | None:
    """Return an editable eye-pair proposal and a face frame, never an identity."""
    scale = min(1.0, 900 / max(image.size))
    small = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    gray = cv2.cvtColor(np.asarray(small.convert("RGB")), cv2.COLOR_RGB2GRAY)
    root = cv2.data.haarcascades
    faces = cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(root + "haarcascade_eye.xml")
    if faces.empty() or eyes.empty():
        return None
    proposals = []
    for x, y, width, height in faces.detectMultiScale(gray, 1.07, 5, minSize=(60, 60)):
        roi = gray[y + height // 10:y + height * 3 // 5, x:x + width]
        found = eyes.detectMultiScale(roi, 1.06, 7, minSize=(max(12, width // 12),) * 2)
        for index, a in enumerate(found):
            for b in found[index + 1:]:
                ax, ay, aw, ah = map(int, a)
                bx, by, bw, bh = map(int, b)
                separation = abs((ax + aw / 2) - (bx + bw / 2))
                if not width * .25 < separation < width * .65:
                    continue
                if abs((ay + ah / 2) - (by + bh / 2)) > height * .10:
                    continue
                if not .6 < aw / bw < 1.65:
                    continue
                box = ((x + min(ax, bx)) / scale, (y + height // 10 + min(ay, by)) / scale,
                       (x + max(ax + aw, bx + bw)) / scale,
                       (y + height // 10 + max(ay + ah, by + bh)) / scale)
                if observed.crop(box).getextrema() != (255, 255):
                    continue
                face = (x / scale, y / scale, (x + width) / scale, (y + height) / scale)
                proposals.append((width * height, box, face))
    if not proposals:
        return None
    _, box, face = max(proposals, key=lambda p: p[0])
    return box, face


def image_regions(path: Path) -> tuple[Image.Image, Guide, list[Region]]:
    image, observed = load_fragment(path)
    guide = read_guide(path, image.size)
    regions = list(guide.regions)
    if guide.auto_eyes and not any(r.role == "eyes" for r in regions):
        proposal = eye_pair(image, observed)
        if proposal:
            regions.append(Region("eyes", proposal[0], "automatic_eye_pair"))
    return image, guide, regions


def collect_details(paths: Iterable[Path], *, check: Callable[[], None] = lambda: None,
                    log: Callable[[str], None] = lambda _text: None) -> tuple[DetailReference, ...]:
    """Select per-part evidence, rather than discarding it by whole-photo quality."""
    best: dict[str, DetailReference] = {}
    for path in dict.fromkeys(paths):
        check()
        if is_generated_image(path):
            log(f"Excluded generated reference: {path.name}")
            continue
        image, _guide, regions = image_regions(path)
        _, observed = load_fragment(path)
        digest = source_digest(path) if regions else ""
        for region in regions:
            if observed.crop(region.box).getextrema() != (255, 255):
                log(f"Skipped masked {region.role} reference in {path.name}; update its facial guide.")
                continue
            patch = image.crop(region.box)
            gray = np.asarray(patch.convert("L"))
            # A labelled but featureless white/black occluder is not useful evidence.
            if gray.std() < 3 or ((gray < 5) | (gray > 250)).mean() > .95:
                log(f"Skipped featureless {region.role} reference in {path.name}")
                continue
            score = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            score += math.log1p(patch.width * patch.height) / 5
            if region.origin == "user":
                score += 100  # An explicit choice outranks a detector guess.
            patch.thumbnail((512, 512), Image.Resampling.LANCZOS)
            candidate = DetailReference(region.role, path, region.box, patch, region.origin, score, digest)
            if region.role not in best or candidate.score > best[region.role].score:
                best[region.role] = candidate
    return tuple(best[role] for role in ROLES if role in best)


def target_face(source: Path | None, placement: dict | None, draft: Image.Image) -> tuple[Box | None, str]:
    """Prefer a source-space hint. Never derive reference pixels from the draft."""
    if source is not None and placement:
        image, guide, regions = image_regions(source)
        face = guide.face_box
        method = "user_face_location"
        if face is None and guide.auto_eyes:
            proposal = eye_pair(image, load_fragment(source)[1])
            if proposal:
                face, method = proposal[1], "source_eye_pair_layout"
        if face is None:
            # Cropped mouths often have no detectable full face. The box provides
            # an approximate scale/centre; the user can correct the whole-face hint.
            for role, span, cy in (("eyes", .76, .38), ("mouth", .50, .76)):
                region = next((r for r in regions if r.role == role), None)
                if region:
                    x0, y0, x1, y1 = region.box
                    width = (x1 - x0) / span
                    height = width * 1.2
                    left, top = (x0 + x1 - width) / 2, (y0 + y1) / 2 - cy * height
                    face = (left, top, left + width, top + height)
                    method = "estimated_from_source_" + role
                    break
        if face is not None:
            sx = placement["source_width"] / image.width
            sy = placement["source_height"] / image.height
            x, y = placement["x"], placement["y"]
            return (x + face[0] * sx, y + face[1] * sy, x + face[2] * sx, y + face[3] * sy), method
    proposal = eye_pair(draft, Image.new("L", draft.size, 255))
    if proposal:
        return proposal[1], "draft_layout_only_not_evidence"
    return None, "face_not_located"


def region_mask(role: str, face: Box, size: tuple[int, int]) -> Image.Image:
    x, y, right, bottom = face
    width, height = right - x, bottom - y
    coordinates = {
        "eyes": (.04, .15, .96, .55),
        "mouth": (.12, .53, .88, .99),
        "hair": (-.40, -.55, 1.40, 1.65),
    }
    a, b, c, d = coordinates[role]
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((x + a * width, y + b * height, x + c * width, y + d * height), fill=255)
    if role == "hair":
        draw.ellipse((x - width * .03, y, right + width * .03, bottom + height * .08), fill=0)
    return mask


def targeted_masks(details: tuple[DetailReference, ...], face: Box, missing: Image.Image
                   ) -> tuple[tuple[DetailReference, ...], tuple[Image.Image, ...], Image.Image]:
    kept, masks = [], []
    union = Image.new("L", missing.size, 0)
    for detail in details:
        mask = ImageChops.multiply(region_mask(detail.role, face, missing.size), missing.convert("L"))
        if np.count_nonzero(np.asarray(mask) > 127) < 32:
            continue
        kept.append(detail)
        masks.append(mask)
        union = ImageChops.lighter(union, mask)
    return tuple(kept), tuple(masks), union
