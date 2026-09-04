"""Generation backend contracts and deterministic classical fallback."""

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
    """Raised when a generation is cancelled."""


class BackendFatalError(RuntimeError):
    """Raised when the selected generation backend cannot continue."""


class GenerationBackend(ABC):
    name = "unknown"
    model_identifier = "unknown"

    def prepare(self) -> None:
        """Load expensive resources before the first generation."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        *,
        cancel_event: threading.Event,
        progress: ProgressCallback | None = None,
    ) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """Release expensive resources."""


class ClassicalFillBackend(GenerationBackend):
    """Fast diagnostic fill; not a substitute for generative reconstruction."""

    name = "OpenCV classical preview"
    model_identifier = "opencv-telea"

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancel_event: threading.Event,
        progress: ProgressCallback | None = None,
    ) -> Image.Image:
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if progress:
            progress(0, 1, "Applying classical preview fill")
        rgb = np.asarray(request.canvas.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.asarray(request.generated_mask.convert("L"))
        filled = cv2.inpaint(bgr, (mask > 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if progress:
            progress(1, 1, "Classical preview complete")
        return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), mode="RGB")
