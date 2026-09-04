"""Local SDXL inpainting/outpainting with IP-Adapter references."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable

from PIL import Image, ImageOps

from bodybuilder.ai.base import (
    BackendFatalError,
    GenerationBackend,
    GenerationCancelled,
    ProgressCallback,
)
from bodybuilder.config import DeviceKind, ModelSettings, SubjectKind
from bodybuilder.core.types import GenerationRequest


class SdxlBackend(GenerationBackend):
    name = "Local SDXL inpainting + IP-Adapter"

    def __init__(
        self,
        *,
        subject_kind: SubjectKind,
        device: DeviceKind,
        models: ModelSettings,
        use_face_adapter: bool,
        log: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self.subject_kind = subject_kind
        self.requested_device = device
        self.models = models
        self.use_face_adapter = use_face_adapter
        self.log = log
        self.model_identifier = models.inpainting_model_id
        self._pipe: object | None = None
        self._torch: object | None = None
        self._device = "cpu"
        self._ip_adapter_loaded = False

    def _resolve_device(self, torch: object) -> str:
        if self.requested_device == DeviceKind.CUDA:
            if not torch.cuda.is_available():
                raise BackendFatalError("CUDA was selected but PyTorch cannot access a CUDA device")
            return "cuda"
        if self.requested_device == DeviceKind.MPS:
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise BackendFatalError("MPS was selected but PyTorch cannot access MPS")
            return "mps"
        if self.requested_device == DeviceKind.CPU:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def prepare(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except Exception as exc:
            raise BackendFatalError(
                'The AI stack is missing. Install it with: python -m pip install -e ".[ai]"'
            ) from exc

        device = self._resolve_device(torch)
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self.log(f"Loading {self.models.inpainting_model_id} on {device}.")
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.models.inpainting_model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)

            try:
                pipe.load_ip_adapter(
                    self.models.ip_adapter_model_id,
                    subfolder=self.models.ip_adapter_subfolder,
                    weight_name=self.models.ip_adapter_weight_name,
                )
                self._ip_adapter_loaded = True
            except Exception as exc:
                self.log(
                    "IP-Adapter could not be loaded; reconstruction will continue with text and "
                    f"source-canvas conditioning only. Reason: {exc}"
                )
                self._ip_adapter_loaded = False
        except Exception as exc:
            raise BackendFatalError(f"Could not initialize the SDXL backend: {exc}") from exc

        self._pipe = pipe
        self._torch = torch
        self._device = device

    def _reference_image(self, request: GenerationRequest) -> Image.Image:
        if request.face_reference_board is None:
            return request.reference_board.convert("RGB")
        general = ImageOps.contain(request.reference_board.convert("RGB"), (768, 384))
        face = ImageOps.contain(request.face_reference_board.convert("RGB"), (768, 384))
        board = Image.new("RGB", (768, 768), "white")
        board.paste(face, ((768 - face.width) // 2, (384 - face.height) // 2))
        board.paste(general, ((768 - general.width) // 2, 384 + (384 - general.height) // 2))
        return board

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancel_event: threading.Event,
        progress: ProgressCallback | None = None,
    ) -> Image.Image:
        self.prepare()
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        assert self._pipe is not None
        assert self._torch is not None
        pipe = self._pipe
        torch = self._torch

        if self._ip_adapter_loaded and hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(request.reference_fidelity)

        generator_device = "cpu" if self._device == "mps" else self._device
        generator = torch.Generator(device=generator_device).manual_seed(request.seed)

        def callback_on_step_end(
            _pipeline: object,
            step: int,
            _timestep: object,
            callback_kwargs: dict[str, object],
        ) -> dict[str, object]:
            if cancel_event.is_set():
                raise GenerationCancelled("Generation cancelled")
            if progress:
                progress(step + 1, request.steps, "Diffusion")
            return callback_kwargs

        arguments: dict[str, object] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image": request.canvas,
            "mask_image": request.generated_mask,
            "strength": request.strength,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "width": request.width,
            "height": request.height,
            "callback_on_step_end": callback_on_step_end,
        }
        if self._ip_adapter_loaded:
            arguments["ip_adapter_image"] = self._reference_image(request)

        try:
            result = pipe(**arguments).images[0]
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise BackendFatalError(f"SDXL generation failed: {exc}") from exc
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        return result.convert("RGB")

    def close(self) -> None:
        pipe = self._pipe
        torch = self._torch
        self._pipe = None
        self._torch = None
        self._ip_adapter_loaded = False
        if pipe is not None:
            del pipe
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
