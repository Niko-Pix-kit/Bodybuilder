"""Optional grouping of unrelated subjects in a mixed input folder."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from bodybuilder.config import DeviceKind
from bodybuilder.core.image_io import load_rgb
from bodybuilder.core.types import ImageAnalysis


@dataclass(slots=True)
class GroupingResult:
    groups: list[list[ImageAnalysis]]
    warning: str = ""


def _connected_components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    remaining = set(range(similarity.shape[0]))
    groups: list[list[int]] = []
    while remaining:
        root = remaining.pop()
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            linked = {index for index in remaining if similarity[current, index] >= threshold}
            remaining -= linked
            component |= linked
            frontier.extend(linked)
        groups.append(sorted(component))
    return groups


def _fallback_embedding(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histograms = []
    for channel, bins, limit in ((0, 24, 180), (1, 16, 256), (2, 16, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], [0, limit]).flatten()
        histograms.append(histogram)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(gray.astype(np.float32) / 255.0)[:16, :16].flatten()
    vector = np.concatenate([*histograms, dct])
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-9)


def _resolve_torch_device(requested: DeviceKind, torch: object) -> str:
    if requested == DeviceKind.CUDA:
        return "cuda"
    if requested == DeviceKind.MPS:
        return "mps"
    if requested == DeviceKind.CPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dinov2_embeddings(
    analyses: list[ImageAnalysis],
    *,
    model_id: str,
    device: DeviceKind,
    log: Callable[[str], None],
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = _resolve_torch_device(device, torch)
    log(f"Loading image grouping model {model_id} on {resolved_device}.")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(resolved_device)
    vectors: list[np.ndarray] = []
    try:
        for analysis in analyses:
            inputs = processor(images=load_rgb(analysis.path), return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
                vector = outputs.last_hidden_state[:, 0, :]
                vector = torch.nn.functional.normalize(vector, dim=-1)
            vectors.append(vector[0].detach().cpu().float().numpy())
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(vectors)


def group_images(
    analyses: list[ImageAnalysis],
    *,
    group_all: bool,
    threshold: float,
    model_id: str,
    device: DeviceKind,
    log: Callable[[str], None] = lambda _message: None,
) -> GroupingResult:
    if group_all or len(analyses) < 2:
        return GroupingResult([analyses])

    warning = ""
    try:
        vectors = _dinov2_embeddings(
            analyses,
            model_id=model_id,
            device=device,
            log=log,
        )
    except Exception as exc:
        warning = (
            "Automatic subject grouping used a lightweight visual fallback because DINOv2 "
            f"was unavailable: {exc}"
        )
        log(warning)
        vectors = np.stack([_fallback_embedding(load_rgb(item.path)) for item in analyses])

    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    similarity = vectors @ vectors.T
    components = _connected_components(similarity, threshold)
    groups = [[analyses[index] for index in component] for component in components]
    groups.sort(key=lambda group: min(str(item.path).casefold() for item in group))
    return GroupingResult(groups=groups, warning=warning)
