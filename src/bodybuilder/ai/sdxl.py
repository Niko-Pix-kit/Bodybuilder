"""Reference-conditioned SDXL. A failed reference adapter is a hard error."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageOps

from bodybuilder.ai.base import (
    BackendFatalError,
    GenerationBackend,
    GenerationCancelled,
    ProgressCallback,
)
from bodybuilder.config import DeviceKind, ModelSettings, SubjectKind
from bodybuilder.core.types import GenerationRequest
from bodybuilder.core.validation import (
    InvalidGenerationError,
    decode_model_output,
    validate_generation,
)


def prepare_reference_images(images: tuple[Image.Image, ...]) -> list[Image.Image]:
    """Avoid CLIP center-cropping away the only visible eye/forehead/hand."""
    return [ImageOps.pad(image.convert("RGB"), (224, 224),
                        method=Image.Resampling.LANCZOS, color=(127, 127, 127)) for image in images]


class SdxlBackend(GenerationBackend):
    name = "SDXL with mandatory multi-image references"

    def __init__(self, *, subject_kind: SubjectKind, device: DeviceKind,
                 models: ModelSettings, use_face_adapter: bool,
                 log: Callable[[str], None] = lambda _message: None) -> None:
        self.subject_kind = subject_kind
        self.requested_device = device
        self.models = models
        self.use_face_adapter = use_face_adapter
        self.log = log
        self.model_identifier = models.inpainting_model_id
        self._pipe: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._full_precision = False
        self.last_generation_metadata: dict[str, Any] = {}

    def _resolve_device(self, torch: Any) -> str:
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        requested = self.requested_device
        if requested == DeviceKind.CUDA and not cuda:
            raise BackendFatalError("CUDA is selected, but this Python environment cannot use it")
        if requested == DeviceKind.MPS and not mps:
            raise BackendFatalError("MPS is selected, but this Python environment cannot use it")
        if requested != DeviceKind.AUTO:
            return requested.value
        return "cuda" if cuda else "mps" if mps else "cpu"

    def prepare(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler
            from transformers import CLIPVisionModelWithProjection
        except (ImportError, OSError) as exc:
            raise BackendFatalError(
                "AI dependencies are missing or incompatible. Install requirements.txt with the "
                "same Python interpreter used to run BodyBuilder. Details: " + str(exc)) from exc
        self._torch = torch
        self._device = self._resolve_device(torch)
        dtype = torch.float16 if self._device == "cuda" and not self._full_precision else torch.float32
        self.log(f"Device: {self._device}; precision: {dtype}. Model download/loading may take time.")
        if self._device == "cpu":
            self.log("No supported GPU selected. Real AI will run on CPU and can be very slow; no fake fill will be substituted.")
        try:
            # ViT-H lives under models/, not sdxl_models/ (the latter contains ViT-bigG).
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.models.ip_adapter_model_id,
                subfolder=self.models.ip_adapter_image_encoder_subfolder,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipe = AutoPipelineForInpainting.from_pretrained(
                self.models.inpainting_model_id, image_encoder=encoder,
                torch_dtype=dtype, use_safetensors=True)
            self._pipe = pipe
            pipe.load_ip_adapter(self.models.ip_adapter_model_id,
                                 subfolder=self.models.ip_adapter_subfolder,
                                 weight_name=self.models.ip_adapter_weight_name,
                                 image_encoder_folder=None)
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            # SDXL's original VAE must upcast its encoder/decoder when using fp16.
            pipe.vae.register_to_config(force_upcast=True)
            pipe.enable_vae_tiling()
            # Load every component BEFORE installing accelerate offload hooks.
            if self._device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self._device)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, ImportError) as exc:
            self.close()
            raise BackendFatalError(
                "Could not load SDXL and its reference-image adapter. Stopped rather than "
                "generating without your photographs. Check the model download, free memory "
                "and dependency installation. Details: " + str(exc)) from exc
        self.log("Reference adapter ready. Each reference is encoded separately, not as a collage.")

    def generate(self, request: GenerationRequest, *, cancel_event: threading.Event,
                 progress: ProgressCallback | None = None) -> Image.Image:
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled")
        if request.generated_mask.getbbox() is None:
            self.last_generation_metadata = {"generation_skipped": "No missing pixels"}
            return request.canvas.convert("RGB").copy()
        if not request.reference_images:
            raise BackendFatalError("No individual reference photographs were supplied")
        if int(request.steps * request.strength) < 1:
            raise BackendFatalError("The current strength/steps would perform no denoising")
        for attempt in range(2):
            self.prepare()
            if cancel_event.is_set():
                raise GenerationCancelled("Cancelled")
            seed = (request.seed + attempt) % (2**32)
            fidelity = request.reference_fidelity if not attempt else min(request.reference_fidelity, 0.5)
            try:
                image = self._generate_once(request, seed, fidelity, cancel_event, progress)
                validate_generation(image, request.canvas, request.generated_mask)
                self.last_generation_metadata = {
                    "effective_seed": seed, "attempts": attempt + 1,
                    "reference_count": len(request.reference_images), "device": self._device,
                    "full_precision_retry": self._full_precision,
                    "reference_fidelity": fidelity,
                    "reference_adapter": self.models.ip_adapter_weight_name,
                }
                return image
            except InvalidGenerationError as exc:
                if attempt:
                    raise BackendFatalError(
                        "AI reconstruction failed twice: " + str(exc) +
                        ". No blank image was saved as a successful result. See bodybuilder.log.") from exc
                self.log(f"Invalid AI output: {exc}. Retrying once in full precision with a new seed.")
                self.close()
                self._full_precision = True
                if cancel_event.is_set():
                    raise GenerationCancelled("Cancelled") from exc
            except RuntimeError as exc:
                if isinstance(exc, (GenerationCancelled, BackendFatalError)):
                    raise
                if "out of memory" in str(exc).lower():
                    raise BackendFatalError(
                        "Insufficient memory for local AI. Close other GPU applications or "
                        "select CPU in Advanced options (slower). No reconstruction was fabricated.") from exc
                raise BackendFatalError(f"SDXL generation failed: {exc}") from exc
        raise BackendFatalError("No valid reconstruction was produced")

    def _generate_once(self, request: GenerationRequest, seed: int, fidelity: float,
                       cancel_event: threading.Event, progress: ProgressCallback | None) -> Image.Image:
        pipe, torch = self._pipe, self._torch
        pipe.set_ip_adapter_scale(fidelity)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def on_step(_pipe: Any, step: int, _time: Any, data: dict[str, Any]) -> dict[str, Any]:
            if cancel_event.is_set():
                raise GenerationCancelled("Cancelled")
            latents = data.get("latents")
            if latents is not None and not bool(torch.isfinite(latents).all()):
                raise InvalidGenerationError("Non-finite diffusion latents")
            if progress:
                progress(step + 1, max(1, int(request.steps * request.strength)), "Reconstructing")
            return data

        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt, negative_prompt=request.negative_prompt,
                image=request.canvas, mask_image=request.generated_mask,
                # Outer list = adapter; inner list = images for that one adapter.
                ip_adapter_image=[prepare_reference_images(request.reference_images)],
                strength=request.strength, num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale, generator=generator,
                width=request.width, height=request.height,
                callback_on_step_end=on_step, output_type="np")
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled")
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged is not None and any(flagged):
            raise BackendFatalError("The model rejected this output; it was not exported")
        if len(result.images) != 1:
            raise InvalidGenerationError("The model returned an unexpected number of images")
        return decode_model_output(result.images[0])

    def close(self) -> None:
        self._pipe = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
