"""Feature-based stitching that never repaints already observed pixels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bodybuilder.core.image_io import load_rgb
from bodybuilder.core.types import StitchResult


def _features(gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None, int]:
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=6000)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors, cv2.NORM_L2
    detector = cv2.AKAZE_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors, cv2.NORM_HAMMING


def _estimate_candidate_to_anchor(
    anchor: Image.Image,
    candidate: Image.Image,
) -> tuple[np.ndarray, int, float] | None:
    anchor_gray = cv2.cvtColor(np.asarray(anchor.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    anchor_points, anchor_desc, norm = _features(anchor_gray)
    candidate_points, candidate_desc, _ = _features(candidate_gray)
    if anchor_desc is None or candidate_desc is None or len(anchor_points) < 8 or len(candidate_points) < 8:
        return None

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(candidate_desc, anchor_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.74 * second.distance]
    if len(good) < 10:
        return None

    candidate_xy = np.float32([candidate_points[match.queryIdx].pt for match in good])
    anchor_xy = np.float32([anchor_points[match.trainIdx].pt for match in good])
    homography, inliers = cv2.findHomography(candidate_xy, anchor_xy, cv2.RANSAC, 4.0)
    if homography is None or inliers is None:
        return None
    inlier_count = int(inliers.ravel().sum())
    inlier_ratio = inlier_count / len(good)
    if inlier_count < 9 or inlier_ratio < 0.45:
        return None

    corners = np.float32(
        [[[0, 0]], [[candidate.width, 0]], [[candidate.width, candidate.height]], [[0, candidate.height]]]
    )
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    source_area = candidate.width * candidate.height
    ratio = projected_area / max(1, source_area)
    if ratio < 0.20 or ratio > 5.0:
        return None
    if not np.isfinite(projected).all():
        return None
    return homography, inlier_count, inlier_ratio


def _merge(
    anchor: Image.Image,
    anchor_mask: Image.Image,
    candidate: Image.Image,
    transform: np.ndarray,
) -> tuple[Image.Image, Image.Image] | None:
    anchor_corners = np.float32(
        [[0, 0], [anchor.width, 0], [anchor.width, anchor.height], [0, anchor.height]]
    )
    candidate_corners = cv2.perspectiveTransform(
        np.float32(
            [[[0, 0]], [[candidate.width, 0]], [[candidate.width, candidate.height]], [[0, candidate.height]]]
        ),
        transform,
    ).reshape(-1, 2)
    all_corners = np.vstack([anchor_corners, candidate_corners])
    minimum = np.floor(all_corners.min(axis=0)).astype(int)
    maximum = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = (maximum - minimum).tolist()
    if width <= 0 or height <= 0 or width * height > 80_000_000:
        return None

    shift = np.array(
        [[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    shifted_transform = shift @ transform
    anchor_x, anchor_y = -minimum[0], -minimum[1]

    candidate_array = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2BGR)
    warped_candidate = cv2.warpPerspective(
        candidate_array,
        shifted_transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    candidate_mask = cv2.warpPerspective(
        np.full((candidate.height, candidate.width), 255, dtype=np.uint8),
        shifted_transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output_mask = np.zeros((height, width), dtype=np.uint8)
    anchor_array = np.asarray(anchor.convert("RGB"))
    anchor_mask_array = np.asarray(anchor_mask.convert("L"))
    output[anchor_y : anchor_y + anchor.height, anchor_x : anchor_x + anchor.width] = anchor_array
    output_mask[anchor_y : anchor_y + anchor.height, anchor_x : anchor_x + anchor.width] = anchor_mask_array

    add_mask = (candidate_mask > 127) & (output_mask <= 127)
    warped_rgb = cv2.cvtColor(warped_candidate, cv2.COLOR_BGR2RGB)
    output[add_mask] = warped_rgb[add_mask]
    output_mask[add_mask] = 255

    new_pixels = int(add_mask.sum())
    if new_pixels < max(64, int(candidate.width * candidate.height * 0.01)):
        return None
    return Image.fromarray(output, mode="RGB"), Image.fromarray(output_mask, mode="L")


def stitch_from_anchor(
    anchor_path: Path,
    candidate_paths: Iterable[Path],
    *,
    max_candidates: int = 6,
    log: Callable[[str], None] = lambda _message: None,
) -> StitchResult:
    image = load_rgb(anchor_path)
    observed = Image.new("L", image.size, 255)
    used = [anchor_path]
    rejected: list[Path] = []

    for candidate_path in list(candidate_paths)[:max_candidates]:
        try:
            candidate = load_rgb(candidate_path)
            estimate = _estimate_candidate_to_anchor(image, candidate)
            if estimate is None:
                rejected.append(candidate_path)
                continue
            transform, inlier_count, inlier_ratio = estimate
            merged = _merge(image, observed, candidate, transform)
            if merged is None:
                rejected.append(candidate_path)
                continue
            image, observed = merged
            used.append(candidate_path)
            log(
                f"Stitched {candidate_path.name}: {inlier_count} inliers, "
                f"{inlier_ratio:.0%} inlier ratio."
            )
        except Exception as exc:
            rejected.append(candidate_path)
            log(f"Skipped overlap candidate {candidate_path.name}: {exc}")
    return StitchResult(image=image, observed_mask=observed, used_paths=used, rejected_paths=rejected)
