"""Optional 2x image enhancement backends."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from PIL import Image

from bodybuilder.config import DeviceKind


class Upscaler(ABC):
    name = "unknown"

    @abstractmethod
    def upscale(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""


class LanczosUpscaler(Upscaler):
    name = "Lanczos 2x"

    def upscale(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize(
            (image.width * 2, image.height * 2),
            Image.Resampling.LANCZOS,
        )


class Swin2SRUpscaler(Upscaler):
    name = "Swin2SR 2x"

    def __init__(
        self,
        *,
        model_id: str,
        device: DeviceKind,
        log: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.log = log
        self._model: object | None = None
        self._processor: object | None = None
        self._torch: object | None = None
        self._device = "cpu"

    def _prepare(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

        if self.requested_device == DeviceKind.CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was selected but is not available")
            device = "cuda"
        elif self.requested_device == DeviceKind.MPS:
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError("MPS was selected but is not available")
            device = "mps"
        elif self.requested_device == DeviceKind.CPU:
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.log(f"Loading {self.model_id} on {device}.")
        self._processor = Swin2SRImageProcessor.from_pretrained(self.model_id)
        self._model = Swin2SRForImageSuperResolution.from_pretrained(self.model_id).eval().to(device)
        self._torch = torch
        self._device = device

    def upscale(self, image: Image.Image) -> Image.Image:
        self._prepare()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        original = image.convert("RGB")
        max_model_edge = 1024
        model_input = original
        if max(original.size) > max_model_edge:
            scale = max_model_edge / max(original.size)
            model_input = original.resize(
                (max(32, round(original.width * scale)), max(32, round(original.height * scale))),
                Image.Resampling.LANCZOS,
            )

        inputs = self._processor(images=model_input, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.inference_mode():
            reconstruction = self._model(pixel_values=pixel_values).reconstruction
        reconstruction = reconstruction.squeeze(0).float().cpu().clamp(0, 1).numpy()
        array = np.moveaxis(reconstruction, 0, -1)
        result = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
        target = (original.width * 2, original.height * 2)
        if result.size != target:
            result = result.resize(target, Image.Resampling.LANCZOS)
        return result

    def close(self) -> None:
        torch = self._torch
        self._model = None
        self._processor = None
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
