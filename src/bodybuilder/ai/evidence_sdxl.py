"""Independent compositions and spatially assigned evidence on the existing SDXL backend."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

from bodybuilder.ai.base import BackendFatalError, GenerationCancelled
from bodybuilder.ai.sdxl import SdxlBackend, prepare_reference_images
from bodybuilder.core.types import GenerationRequest
from bodybuilder.core.validation import InvalidGenerationError, decode_model_output


@dataclass
class EvidenceRequest(GenerationRequest):
    free_composition: bool = False
    reference_masks: tuple[Image.Image, ...] = ()


def reference_conditioning(pipe: Any, request: EvidenceRequest, fidelity: float) -> dict[str, Any]:
    """Use the documented one-adapter / multiple-image mask layout."""
    if request.reference_masks:
        from diffusers.image_processor import IPAdapterMaskProcessor
        if len(request.reference_masks) != len(request.reference_images):
            raise ValueError("Each reference must have exactly one target-region mask")
        if any(mask.size != (request.width, request.height) for mask in request.reference_masks):
            raise ValueError("Reference masks must match the generation canvas")
        masks = IPAdapterMaskProcessor().preprocess(list(request.reference_masks),
                                                   height=request.height, width=request.width)
        pipe.set_ip_adapter_scale([[fidelity] * len(request.reference_images)])
        return {"cross_attention_kwargs": {"ip_adapter_masks": [masks.reshape(
            1, len(request.reference_masks), request.height, request.width)]}}
    if request.free_composition:
        # The documented style/appearance block excludes the down block that
        # strongly copies layout. Original details are reinforced in regional passes.
        pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, fidelity, 0.0]}})
    else:
        pipe.set_ip_adapter_scale(fidelity)
    return {}


class EvidenceSdxlBackend(SdxlBackend):
    """Keep validated initialization, cancellation and bounded numerical retry."""
    def _generate_once(self, request: EvidenceRequest, seed: int, fidelity: float,
                       cancel_event: threading.Event, progress) -> Image.Image:
        pipe, torch = self._pipe, self._torch
        kwargs = reference_conditioning(pipe, request, fidelity)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def on_step(_pipe, step, _time, data):
            if cancel_event.is_set():
                raise GenerationCancelled("Cancelled")
            latents = data.get("latents")
            if latents is not None and not bool(torch.isfinite(latents).all()):
                raise InvalidGenerationError("Non-finite diffusion latents")
            if progress:
                progress(step + 1, max(1, int(request.steps * request.strength)), "Reconstructing")
            return data

        with torch.inference_mode():
            result = pipe(prompt=request.prompt, negative_prompt=request.negative_prompt,
                image=request.canvas, mask_image=request.generated_mask,
                ip_adapter_image=[prepare_reference_images(request.reference_images)],
                strength=request.strength, num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale, generator=generator,
                width=request.width, height=request.height, callback_on_step_end=on_step,
                output_type="np", **kwargs)
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled")
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged is not None and any(flagged):
            raise BackendFatalError("The model rejected this output; it was not exported")
        if len(result.images) != 1:
            raise InvalidGenerationError("The model returned an unexpected number of images")
        return decode_model_output(result.images[0])
