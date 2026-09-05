"""Conservative same-view hole repair, independent of object category.

A transform is accepted only with distributed feature inliers and photometric
agreement. Different poses remain generative references, never pasted as evidence.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from bodybuilder.core.subject_evidence import SourceView


def estimate_overlap(anchor: SourceView, donor: SourceView):
    a = cv2.cvtColor(np.asarray(anchor.image), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(donor.image), cv2.COLOR_RGB2GRAY)
    am = np.asarray(anchor.observed).copy()
    bm = np.asarray(donor.observed).copy()
    # Restrict feature matching to the proposed subject, not a repeated background.
    for mask, box in ((am, anchor.subject_box), (bm, donor.subject_box)):
        if box is not None:
            subject = np.zeros_like(mask)
            x0, y0, x1, y1 = box
            subject[y0:y1, x0:x1] = 255
            mask &= subject
    detector = cv2.SIFT_create(nfeatures=2000)
    ak, ad = detector.detectAndCompute(a, am)
    bk, bd = detector.detectAndCompute(b, bm)
    if ad is None or bd is None or min(len(ak), len(bk)) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(bd, ad, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.68 * pair[1].distance]
    if len(good) < 12:
        return None
    source = np.float32([bk[m.queryIdx].pt for m in good])
    target = np.float32([ak[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 12 or keep.mean() < 0.75:
        return None
    # Reject extrapolation from a tiny single patch or almost-collinear matches.
    hull = cv2.convexHull(target[keep])
    if abs(cv2.contourArea(hull)) < 0.025 * max(1, np.count_nonzero(am)):
        return None
    corners = np.float32([[[0, 0]], [[b.shape[1], 0]], [[b.shape[1], b.shape[0]]], [[0, b.shape[0]]]])
    projected = cv2.perspectiveTransform(corners, matrix)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return None
    ratio = abs(cv2.contourArea(projected)) / (b.shape[0] * b.shape[1])
    if not 0.15 <= ratio <= 6:
        return None
    warped = cv2.warpPerspective(np.asarray(donor.image), matrix, anchor.image.size)
    warped_mask = cv2.warpPerspective(bm, matrix, anchor.image.size, flags=cv2.INTER_NEAREST)
    overlap = (am > 127) & (warped_mask > 127)
    if overlap.sum() < 256:
        return None
    difference = np.abs(np.asarray(anchor.image, dtype=np.float32) - warped.astype(np.float32)).mean(axis=2)
    # Require agreement over most of the overlap, not just matched keypoints.
    if np.median(difference[overlap]) > 18 or np.quantile(difference[overlap], 0.9) > 45:
        return None
    return matrix, {"donor": str(donor.path), "homography": matrix.tolist(),
                    "inliers": int(keep.sum()),
                    "notice": "Geometrically resampled source evidence; not newly generated pixels"}


def register_observed(anchor: SourceView, donor: SourceView):
    estimate = estimate_overlap(anchor, donor)
    if estimate is None:
        return None
    matrix, record = estimate
    warped = cv2.warpPerspective(np.asarray(donor.image), matrix, anchor.image.size)
    mask = cv2.warpPerspective(np.asarray(donor.observed), matrix, anchor.image.size,
                               flags=cv2.INTER_NEAREST)
    reliable = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 127
    holes = (np.asarray(anchor.observed) < 128) & reliable
    if holes.sum() < 16:
        return None
    record["pixels_recovered"] = int(holes.sum())
    return warped, holes, record


def assemble_overlap(anchor: SourceView, donor: SourceView):
    """Union canvas for real overlapping crops; no guessed alignment or inpainting."""
    estimate = estimate_overlap(anchor, donor)
    if estimate is None:
        return None
    matrix, record = estimate
    corners = np.float32([[[0, 0]], [[donor.image.width, 0]],
                           [[donor.image.width, donor.image.height]], [[0, donor.image.height]]])
    warped_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    all_corners = np.vstack((warped_corners, [[0, 0], [anchor.image.width, anchor.image.height]]))
    low = np.floor(all_corners.min(axis=0)).astype(int)
    high = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = (high - low).tolist()
    if min(width, height) < 8 or width * height > 4_000_000:
        return None
    shift = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], dtype=np.float64)
    pixels = cv2.warpPerspective(np.asarray(donor.image), shift @ matrix, (width, height))
    mask = cv2.warpPerspective(np.asarray(donor.observed), shift @ matrix, (width, height),
                               flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    image = Image.fromarray(pixels)
    observed = Image.fromarray(mask)
    x, y = int(-low[0]), int(-low[1])
    image.paste(anchor.image, (x, y), anchor.observed)
    observed.paste(255, (x, y, x + anchor.image.width, y + anchor.image.height), anchor.observed)
    added = np.count_nonzero(np.asarray(observed)) - np.count_nonzero(np.asarray(anchor.observed))
    if added < 64:
        return None
    image.paste((127, 127, 127), mask=Image.eval(observed, lambda value: 255 - value))
    record.update({"anchor": str(anchor.path), "anchor_offset": [x, y],
                   "pixels_added": int(added), "canvas_size": [width, height]})
    return image, observed, record


def repair_from_overlaps(anchor: SourceView, donors: list[SourceView], check_cancelled=lambda: None):
    image, observed = anchor.image.copy(), anchor.observed.copy()
    records = []
    for donor in donors:
        check_cancelled()
        if donor.path == anchor.path:
            continue
        # Always estimate against the original evidence, not previously synthesized content.
        result = register_observed(anchor, donor)
        if result is None:
            continue
        warped, holes, record = result
        holes &= np.asarray(observed) < 128
        if not holes.any():
            continue
        mask = Image.fromarray(holes.astype(np.uint8) * 255)
        image.paste(Image.fromarray(warped), mask=mask)
        observed.paste(255, mask=mask)
        record["pixels_recovered"] = int(holes.sum())
        records.append(record)
    return image, observed, records
