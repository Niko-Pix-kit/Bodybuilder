"""Two-stage reconstruction with spatially routed eyes/mouth/hair references."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import replace
from typing import Protocol

from PIL import Image, ImageChops, ImageFilter

from bodybuilder.ai.base import GenerationCancelled, ProgressCallback
from bodybuilder.core.evidence import collect_details, target_face, targeted_masks
from bodybuilder.core.types import GenerationRequest


class DetailEngine(Protocol):
    last_generation_metadata: dict

    def log(self, message: str) -> None: ...

    def _generate_validated(self, request: GenerationRequest, *, cancel_event: threading.Event,
                            progress: ProgressCallback | None = None) -> Image.Image: ...


def generate_with_details(engine: DetailEngine, request: GenerationRequest, *,
                          cancel_event: threading.Event,
                          progress: ProgressCallback | None = None) -> Image.Image:
    def check() -> None:
        if cancel_event.is_set():
            raise GenerationCancelled("Cancelled during facial evidence processing")

    details = collect_details(request.evidence_paths, check=check, log=engine.log)
    check()
    has_eyes = any(detail.role == "eyes" for detail in details)
    prompt, negative = request.prompt, request.negative_prompt
    if has_eyes:
        prompt += "; natural unobstructed eyes and eyelids matching the eye reference, no added eyewear"
        negative += ", sunglasses, tinted lenses, added glasses, goggles, eye-covering accessories"
    # Give small but informative patches a place even when a full-body photo has a
    # higher whole-image quality score. They are routed spatially in the next pass.
    images = request.reference_images
    if details and images:
        images = (images[0], *(d.image for d in details), *images[1:])[:16]
    draft = engine._generate_validated(replace(request, prompt=prompt, negative_prompt=negative,
                                               reference_images=images),
                                        cancel_event=cancel_event, progress=progress)
    first_metadata = deepcopy(engine.last_generation_metadata)
    report = {"evidence": [d.metadata() for d in details],
              "identity_verified": False, "eyewear_constraint": has_eyes,
              "notice": "Regional image guidance is not a guarantee of identity or exact recovered pixels."}
    if not details:
        report["status"] = "no_regional_evidence"
        engine.log("No usable facial details found. Use 'Facial references' to mark visible eyes or mouth.")
    else:
        face, location_method = target_face(request.source_path, request.source_placement, draft)
        report.update({"face_location_method": location_method, "face_box": face})
        if face is None:
            report["status"] = "face_location_needed"
            engine.log("FACIAL GUIDANCE NOT APPLIED: face position is uncertain. In 'Facial references', "
                       "mark 'Face location' on the source canvas. A general completion is kept for review.")
        else:
            selected, masks, union = targeted_masks(details, face, request.generated_mask)
            if not selected:
                report["status"] = "facial_regions_already_observed"
                engine.log("Facial reference regions are already visible and will not be repainted. "
                           "Paint an opaque obstruction as missing to reconstruct it.")
            else:
                check()
                roles = ", ".join(d.role for d in selected)
                engine.log(f"Refining missing facial details from original references: {roles} ({location_method}).")
                if progress:
                    progress(0, request.steps, "Refining " + roles)
                local_request = replace(request, canvas=draft, generated_mask=union,
                    prompt=prompt + "; consistent facial details, one face, coherent skin and lighting",
                    negative_prompt=negative, seed=(request.seed + 7919) % (2**32),
                    reference_images=tuple(d.image for d in selected), reference_masks=masks,
                    reference_fidelity=max(.75, request.reference_fidelity), strength=.90)
                refined = engine._generate_validated(local_request, cancel_event=cancel_event,
                                                       progress=progress)
                # Feather only inside the editable region; never alter original
                # evidence or any pixels outside this facial refinement pass.
                feather = union.filter(ImageFilter.GaussianBlur(3))
                blend = ImageChops.multiply(feather, union)
                draft = Image.composite(refined, draft, blend)
                report.update({"status": "regional_refinement_applied", "refined_roles": roles,
                               "target_mask_boxes": [mask.getbbox() for mask in masks],
                               "generation": deepcopy(engine.last_generation_metadata)})
    # Even a misbehaving backend must not modify the known regions.
    draft = Image.composite(request.canvas, draft, ImageChops.invert(request.generated_mask.convert("L")))
    engine.last_generation_metadata = {**first_metadata, "facial_guidance": report}
    return draft
