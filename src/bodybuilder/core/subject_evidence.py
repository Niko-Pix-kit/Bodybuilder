"""Subject-level evidence, complementary part selection and bounded references."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bodybuilder.ai.subject_vision import Region

_PERSON = {"person", "woman", "man", "girl", "boy", "face", "human", "lady", "people"}
_STOP = set("a an the of with on in at and its is are this that some pair visible close up view photo image part front back side left right black white red blue green brown small large wearing standing sitting".split())
# Localization hints only; selection and refinement are shared with arbitrary objects.
PERSON_PARTS = ("eyes", "mouth", "hair", "nose", "hands", "feet")


def subject_key(label: str) -> str:
    words = re.findall(r"[a-z]+", label.lower())
    if label.strip().lower() in _PERSON or set(words) & (_PERSON - {"face"}):
        return "person"
    return " ".join(word for word in words if word not in _STOP)


def part_key(label: str) -> str:
    words = re.findall(r"[a-z]+", label.lower())
    # Keep only explicit short region names. Long descriptions are never treated as facts.
    for key, aliases in (("eyes", {"eye", "eyes"}), ("mouth", {"mouth", "lip", "lips"}),
                         ("hair", {"hair"}), ("nose", {"nose"})):
        if set(words) & aliases:
            return key
    kept = [word for word in words if word not in _STOP]
    return " ".join(kept) if 1 <= len(kept) <= 4 else ""


def intersection(a, b) -> tuple[int, int, int, int]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


@dataclass
class SourceView:
    path: Path
    image: Image.Image
    observed: Image.Image
    detections: list[Region]
    subject_box: tuple[int, int, int, int] | None = None


@dataclass
class PartEvidence:
    name: str
    source: Path
    box: tuple[int, int, int, int]
    image: Image.Image
    score: float

    def record(self) -> dict:
        return {"part": self.name, "source": str(self.source), "box": self.box,
                "selection_score": round(self.score, 4),
                "score_meaning": "visibility/detail heuristic, not probability of identity correctness"}


@dataclass
class SubjectEvidence:
    label: str
    views: list[SourceView]
    parts: list[PartEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    accessory_labels: set[str] = field(default_factory=set)
    assemblies: list[tuple[Image.Image, dict]] = field(default_factory=list)

    def report(self) -> dict:
        return {"common_subject_candidate": self.label,
                "notice": "Machine-localized evidence, not verified identity or a recovered 3D model.",
                "sources": [{"path": str(v.path), "subject_box": v.subject_box} for v in self.views],
                "parts": [p.record() for p in self.parts], "warnings": self.warnings,
                "registered_assemblies": [record for _, record in self.assemblies]}

    def references(self, limit: int = 16) -> tuple[tuple[Image.Image, ...], list[dict]]:
        """Reserve capacity for distinct parts, not sixteen nearly identical whole crops."""
        if limit < 1:
            raise ValueError("Reference limit must be positive")
        images, records = [], []
        chosen_parts = self.parts[:min(8, limit // 2)]
        for image, record in self.assemblies[:min(2, limit // 4)]:
            fitted = image.copy()
            fitted.thumbnail((768, 768))
            images.append(fitted)
            records.append({"kind": "registered_source_fragments", **record})
        slots = limit - len(chosen_parts) - len(images)
        for view in self.views[:slots]:
            box = view.subject_box or view.observed.getbbox()
            image = view.image.crop(box)
            image.thumbnail((768, 768))
            images.append(image)
            records.append({"kind": "subject_crop", "source": str(view.path), "box": box})
        for part in chosen_parts:
            images.append(part.image)
            records.append({"kind": "visible_part", **part.record()})
        if not images:
            raise ValueError("No subject evidence available")
        return tuple(images), records


def choose_common_subject(views: list[SourceView], hint: str = "") -> str:
    if hint.strip():
        cleaned = hint.strip().lower()
        if len(cleaned) > 80 or any(c in cleaned for c in "<>\n"):
            raise ValueError("Subject hint must be a short object name, for example person or bicycle")
        return cleaned
    support: dict[str, set[int]] = {}
    sizes: dict[str, float] = {}
    for index, view in enumerate(views):
        for region in view.detections:
            key = subject_key(region.label)
            if not key:
                continue
            support.setdefault(key, set()).add(index)
            sizes[key] = sizes.get(key, 0.0) + region.area / (view.image.width * view.image.height)
    ranked = sorted(support, key=lambda key: (len(support[key]), sizes[key]), reverse=True)
    minimum = 1 if len(views) == 1 else 2
    if not ranked or len(support[ranked[0]]) < minimum:
        raise ValueError("Cannot reliably select a common subject. Enter its name in Subject hint "
                         "under Advanced options. Do not mix unrelated subjects in one folder.")
    if len(ranked) > 1 and len(support[ranked[0]]) == len(support[ranked[1]]) and sizes[ranked[1]] > 0.8 * sizes[ranked[0]]:
        raise ValueError("Several common objects are plausible: " + ", ".join(ranked[:3]) +
                         ". Enter the intended object in Subject hint.")
    return ranked[0]


def visible_part(view: SourceView, region: Region, name: str) -> PartEvidence | None:
    box = intersection(region.box, view.subject_box or (0, 0, *view.image.size))
    if min(box[2] - box[0], box[3] - box[1]) < 12:
        return None
    mask = np.asarray(view.observed.crop(box)) > 127
    if mask.mean() < 0.90:
        return None
    crop = view.image.crop(box)
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    valid = gray[mask]
    # An opaque white erasure must not become a mouth reference. This is a crop
    # eligibility rule, never automatic erasure of white pixels in an original.
    if valid.std() < 5 or ((valid > 248) | (valid < 5)).mean() > 0.70:
        return None
    detail = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = float(mask.mean()) * np.log1p(detail) * np.log1p(mask.sum())
    crop.thumbnail((512, 512))
    return PartEvidence(name, view.path, box, crop, float(score))


def build_evidence(views: list[SourceView], vision, *, hint: str = "", log=lambda _text: None) -> SubjectEvidence:
    label = choose_common_subject(views, hint)
    evidence = SubjectEvidence(label, views)
    best: dict[str, PartEvidence] = {}
    log(f"Common subject candidate: {label}. Comparing visible parts across all source photographs.")
    for view in views:
        vision.check_cancelled()
        matches = [r for r in view.detections if subject_key(r.label) == subject_key(label)]
        if not matches:
            matches = vision.locate(view.image, label)
        matches.sort(key=lambda region: region.area, reverse=True)
        if len(matches) > 1 and matches[1].area > matches[0].area * 0.5:
            raise ValueError(f"Several {label} instances in {view.path.name}; isolate the intended subject first")
        if matches:
            view.subject_box = matches[0].box
        else:
            evidence.warnings.append(f"Subject not localized in {view.path.name}; retained as an unlocalized fragment")
        regions = vision.describe_regions(view.image)
        evidence.accessory_labels.update(r.label for r in (*view.detections, *regions))
        if subject_key(label) == "person":
            # Targeted part queries supplement generic region captions for small facial details.
            for name in PERSON_PARTS:
                regions.extend(Region(name, r.box) for r in vision.locate(view.image, name))
        for region in regions:
            name = part_key(region.label)
            if not name or subject_key(name) == subject_key(label):
                continue
            part = visible_part(view, region, name)
            if part is not None and (name not in best or part.score > best[name].score):
                best[name] = part
    priority = {name: i for i, name in enumerate(PERSON_PARTS)}
    evidence.parts = sorted(best.values(), key=lambda p: (priority.get(p.name, 10), -p.score))[:16]
    if not evidence.parts:
        evidence.warnings.append("No detailed parts could be localized reliably; only whole-fragment references are available")
    if len(views) > 8:
        evidence.warnings.append("All sources were analyzed; each diffusion pass uses a bounded reference subset recorded in metadata")
    evidence.warnings.append("Common-category detection does not prove that every fragment belongs to the same physical instance. Review subject_evidence.json.")
    evidence.warnings.append("Unseen surfaces and body parts are hypotheses. A bounding-box check is not proof of complete anatomy or geometry.")
    return evidence


def full_subject_prompt(evidence: SubjectEvidence, index: int, description: str = "") -> tuple[str, str]:
    person = subject_key(evidence.label) == "person"
    if person:
        views = ("standing naturally, front view", "standing naturally, three-quarter view",
                 "standing naturally, side view", "seated naturally, whole body visible")
        framing = "full-length clothed person, entire head, hair, hands, legs and both feet in frame"
    else:
        views = ("front view", "three-quarter view", "side view", "elevated three-quarter view")
        framing = f"one complete {evidence.label}, entire outer silhouette and all components in frame"
    prompt = (f"Realistic photograph of {framing}, {views[index % len(views)]}, "
              "camera pulled back, subject centered with generous empty space on every side, "
              "new coherent composition, plain neutral background. Same subject and visible details "
              "as the reference photographs, no additional accessories. " + description.strip())
    negative = ("cropped subject, close-up, selfie, extreme perspective, cut-off head, cut-off feet, "
                "body out of frame, collage, split image, duplicate subject, extra parts, text, watermark")
    if person and any(part.name == "eyes" for part in evidence.parts):
        if not any(re.search(r"\b(sunglasses|glasses|goggles|spectacles)\b", label) for label in evidence.accessory_labels):
            prompt += " Unobstructed eyes matching the visible eye references."
            negative += ", sunglasses, eyeglasses, goggles, spectacles"
    return prompt, negative


def check_whole_framing(regions: list[Region], size: tuple[int, int]) -> str | None:
    """Conservative geometry gate, not a guarantee that unseen anatomy is correct."""
    if not regions:
        return "The generated subject could not be localized"
    regions = sorted(regions, key=lambda region: region.area, reverse=True)
    if len(regions) > 1 and regions[1].area > regions[0].area * 0.35:
        return "More than one substantial subject was generated"
    left, top, right, bottom = regions[0].box
    mx, my = max(2, size[0] * 0.015), max(2, size[1] * 0.015)
    if left <= mx or top <= my or right >= size[0] - mx or bottom >= size[1] - my:
        return "The generated subject touches an image edge; it may be cropped"
    return None
