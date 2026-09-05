"""Optional 2x enhancement; the pipeline restores observed details afterwards."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from bodybuilder.config import DeviceKind


class Upscaler(ABC):
    name = "unknown"

    @abstractmethod
    def upscale(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """Optional resource cleanup."""
        return None


class LanczosUpscaler(Upscaler):
    name = "Lanczos 2x"

    def upscale(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)


class Swin2SRUpscaler(Upscaler):
    name = "Swin2SR 2x"

    def __init__(self, *, model_id: str, device: DeviceKind,
                 log: Callable[[str], None] = lambda _message: None) -> None:
        self.model_id, self.requested_device, self.log = model_id, device, log
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"

    def _prepare(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        device = self.requested_device.value
        if device == "auto":
            device = "cuda" if cuda else "mps" if mps else "cpu"
        if (device == "cuda" and not cuda) or (device == "mps" and not mps):
            raise RuntimeError(f"Selected enhancement device is unavailable: {device}")
        self.log(f"Loading enhancement model on {device}")
        self._processor = Swin2SRImageProcessor.from_pretrained(self.model_id)
        self._model = Swin2SRForImageSuperResolution.from_pretrained(self.model_id).eval().to(device)
        self._torch, self._device = torch, device

    def upscale(self, image: Image.Image) -> Image.Image:
        self._prepare()
        original = image.convert("RGB")
        model_input = original.copy()
        model_input.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        inputs = self._processor(images=model_input, return_tensors="pt")
        with self._torch.inference_mode():
            tensor = self._model(pixel_values=inputs["pixel_values"].to(self._device)).reconstruction
        array = tensor.squeeze(0).float().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("Enhancement produced non-finite pixels")
        rgb = np.moveaxis(np.clip(array, 0, 1), 0, -1)
        # Remove processor padding before resizing, rather than distorting the whole image.
        rgb = rgb[:model_input.height * 2, :model_input.width * 2]
        result = Image.fromarray((rgb * 255).round().astype(np.uint8))
        return result.resize((original.width * 2, original.height * 2), Image.Resampling.LANCZOS)

    def close(self) -> None:
        self._model, self._processor = None, None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
