"""Image inspection, face crop discovery, and reference-board construction."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from bodybuilder.core.image_io import load_rgb
from bodybuilder.core.quality import measure_quality
from bodybuilder.core.types import FaceBox, ImageAnalysis

_FACE_DETECTOR_LOCK = threading.Lock()
_FACE_DETECTORS: tuple[cv2.CascadeClassifier, ...] | None = None


def _face_detectors() -> tuple[cv2.CascadeClassifier, ...]:
    global _FACE_DETECTORS
    with _FACE_DETECTOR_LOCK:
        if _FACE_DETECTORS is None:
            cascade_root = Path(cv2.data.haarcascades)
            names = (
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            )
            detectors: list[cv2.CascadeClassifier] = []
            for name in names:
                detector = cv2.CascadeClassifier(str(cascade_root / name))
                if not detector.empty():
                    detectors.append(detector)
            _FACE_DETECTORS = tuple(detectors)
    return _FACE_DETECTORS


def detect_faces(image: Image.Image) -> tuple[FaceBox, ...]:
    detectors = _face_detectors()
    if not detectors:
        return ()

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 1400.0 / max(gray.shape))
    resized = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )

    boxes: list[FaceBox] = []
    for detector in detectors:
        found = detector.detectMultiScale(
            resized,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(28, 28),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, width, height in found:
            boxes.append(
                FaceBox(
                    x=round(x / scale),
                    y=round(y / scale),
                    width=round(width / scale),
                    height=round(height / scale),
                )
            )

    boxes.sort(key=lambda box: box.width * box.height, reverse=True)
    deduplicated: list[FaceBox] = []
    for box in boxes:
        if not any(_intersection_over_union(box, existing) > 0.45 for existing in deduplicated):
            deduplicated.append(box)
    return tuple(deduplicated[:8])


def _intersection_over_union(first: FaceBox, second: FaceBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / max(1, union)


def analyze_image(path: Path) -> ImageAnalysis:
    image = load_rgb(path)
    quality = measure_quality(image)
    warnings: list[str] = []
    if min(image.size) < 256:
        warnings.append("Very small source image")
    if quality.blur < 0.25:
        warnings.append("Strong blur or very low local detail")
    if quality.exposure < 0.35:
        warnings.append("Severe underexposure, overexposure, or clipping")
    faces = detect_faces(image)
    return ImageAnalysis(
        path=path,
        width=image.width,
        height=image.height,
        file_size=path.stat().st_size,
        blur_score=quality.blur,
        exposure_score=quality.exposure,
        detail_score=quality.detail,
        resolution_score=quality.resolution,
        quality_score=quality.overall,
        faces=faces,
        warnings=tuple(warnings),
    )


def analyze_paths(
    paths: Iterable[Path],
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[ImageAnalysis], list[dict[str, str]]]:
    all_paths = list(paths)
    analyses: list[ImageAnalysis] = []
    failures: list[dict[str, str]] = []
    total = len(all_paths)
    for index, path in enumerate(all_paths, start=1):
        if cancelled and cancelled():
            break
        if progress:
            progress(index - 1, total, f"Analyzing {path.name}")
        try:
            analyses.append(analyze_image(path))
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    if progress:
        progress(len(analyses) + len(failures), total, "Analysis complete")
    return analyses, failures


def crop_face(image: Image.Image, face: FaceBox, *, padding: float = 0.35) -> Image.Image:
    pad_x = round(face.width * padding)
    pad_y = round(face.height * padding)
    left = max(0, face.x - pad_x)
    top = max(0, face.y - pad_y)
    right = min(image.width, face.x + face.width + pad_x)
    bottom = min(image.height, face.y + face.height + pad_y)
    return image.crop((left, top, right, bottom)).convert("RGB")


def best_face_crops(analyses: Iterable[ImageAnalysis], *, limit: int = 8) -> list[Image.Image]:
    candidates: list[tuple[float, Image.Image]] = []
    for analysis in analyses:
        if not analysis.faces:
            continue
        image = load_rgb(analysis.path)
        for face in analysis.faces[:2]:
            area_ratio = (face.width * face.height) / max(1, image.width * image.height)
            score = analysis.quality_score * 0.65 + min(1.0, area_ratio * 5.0) * 0.35
            candidates.append((score, crop_face(image, face)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [image for _, image in candidates[:limit]]


def make_reference_board(
    images: Iterable[Image.Image],
    *,
    max_images: int = 16,
    tile_size: int = 224,
    title: str | None = None,
) -> Image.Image:
    selected = [image.convert("RGB") for image in images][:max_images]
    if not selected:
        raise ValueError("Cannot build a reference board without images")

    count = len(selected)
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    title_height = 36 if title else 0
    board = Image.new("RGB", (columns * tile_size, rows * tile_size + title_height), "white")
    if title:
        draw = ImageDraw.Draw(board)
        draw.text((12, 11), title, fill="black", font=ImageFont.load_default())

    for index, image in enumerate(selected):
        fitted = ImageOps.contain(image, (tile_size - 12, tile_size - 12), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), (236, 236, 236))
        tile.paste(fitted, ((tile_size - fitted.width) // 2, (tile_size - fitted.height) // 2))
        column = index % columns
        row = index // columns
        board.paste(tile, (column * tile_size, title_height + row * tile_size))
    return board
