"""Backend contracts and an explicitly non-generative developer preview."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from bodybuilder.core.types import GenerationRequest

ProgressCallback = Callable[[int, int, str], None]


class GenerationCancelled(RuntimeError):
    """The user cancelled generation."""


class BackendFatalError(RuntimeError):
    """The backend cannot continue safely."""


class GenerationBackend(ABC):
    name = "unknown"
    model_identifier = "unknown"

    def prepare(self) -> None:
        """Optional resource acquisition for stateful backends."""
        return None

    @abstractmethod
    def generate(self, request: GenerationRequest, *, cancel_event: threading.Event,
                 progress: ProgressCallback | None = None) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup for stateful backends."""
        return None


class ClassicalFillBackend(GenerationBackend):
    """Developer diagnostics only; never a fallback for failed AI generation."""

    name = "OpenCV diagnostic fill (not reconstruction)"
    model_identifier = "opencv-telea"

    def generate(self, request: GenerationRequest, *, cancel_event: threading.Event,
                 progress: ProgressCallback | None = None) -> Image.Image:
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled")
        mask = (np.asarray(request.generated_mask.convert("L")) > 127).astype(np.uint8) * 255
        pixels = cv2.cvtColor(np.asarray(request.canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(pixels, mask, 5, cv2.INPAINT_TELEA)
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled")
        if progress:
            progress(1, 1, "Diagnostic fill only")
        return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
