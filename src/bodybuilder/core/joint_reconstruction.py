"""Joint subject reconstruction: originals -> evidence -> whole views -> local details.

Source repair and free composition have different coordinate systems. Generated
views are never passed off as recovered observations or fed back as source evidence.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from bodybuilder.ai.base import BackendFatalError, GenerationCancelled
from bodybuilder.ai.evidence_sdxl import EvidenceRequest, EvidenceSdxlBackend
from bodybuilder.ai.subject_vision import SubjectVision
from bodybuilder.config import BackendKind, SubjectKind, VariantFrame
from bodybuilder.core.canvas import prepare_outpaint_canvas, prepare_variant_canvas, preserve_observed_pixels
from bodybuilder.core.fragment_registration import assemble_overlap, repair_from_overlaps
from bodybuilder.core.image_io import ensure_unique_path, load_fragment, safe_stem, save_png, write_json
from bodybuilder.core.pipeline import (
    PipelineCancelled,
    PipelineRunResult,
    ReconstructionPipeline,
    analyze_input_folder,
    environment_report,
)
from bodybuilder.core.subject_evidence import (
    SourceView,
    build_evidence,
    check_whole_framing,
    full_subject_prompt,
    subject_key,
)


class EvidenceConsistencyError(BackendFatalError):
    """A semantic check found an unsupported change, not a numerical failure."""


class JointReconstructionPipeline(ReconstructionPipeline):
    def __init__(self, config, *, complete_subject=True, subject_hint="", **kwargs):
        super().__init__(config, **kwargs)
        self.complete_subject = complete_subject
        self.subject_hint = subject_hint
        self.vision = SubjectVision(self.cancel_event, self._log)
        self.evidence = None
        self.backend = None

    def run(self) -> PipelineRunResult:
        if self.config.backend != BackendKind.SDXL:
            raise ValueError("Joint subject reconstruction requires the real AI backend")
        self._validate_paths()
        name = safe_stem(self.config.run_name) if self.config.run_name else datetime.now().strftime("BodyBuilder_%Y%m%d_%H%M%S")
        self.run_dir = ensure_unique_path(self.config.output_dir / name)
        self.run_dir.mkdir(parents=True)
        self.logger = logging.getLogger(f"bodybuilder.joint.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.run_dir / "bodybuilder.log", encoding="utf-8")
        self.logger.addHandler(handler)
        result = PipelineRunResult(self.run_dir, self.run_dir / "run_manifest.json")
        manifest = {"status": "running", "workflow": "whole_subject" if self.complete_subject else "joint_source_repair",
                    "config": self.config.to_dict(), "environment": environment_report(),
                    "sources": [], "outputs": [], "errors": [], "warnings": []}
        try:
            write_json(result.manifest_path, manifest)
            self.callbacks.progress(0, 0, "Finding the common subject and complementary visible parts...")
            analysis = analyze_input_folder(self.config, callbacks=self.callbacks, cancel_event=self.cancel_event)
            result.errors.extend(analysis.failures)
            manifest["errors"].extend(analysis.failures)
            views = []
            for item in analysis.analyses:
                self._check_cancelled()
                with item.path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                image, observed = load_fragment(item.path)
                image.thumbnail((1024, 1024))
                observed = observed.resize(image.size, Image.Resampling.NEAREST)
                self.callbacks.progress(0, 0, f"Inspecting original fragment: {item.path.name}")
                views.append(SourceView(item.path, image, observed, self.vision.detect(image)))
                manifest["sources"].append({**item.to_dict(), "sha256": digest,
                                            "evidence_working_size": image.size})
            hint = self.subject_hint or ("person" if self.config.subject_kind == SubjectKind.PERSON else "")
            self.evidence = build_evidence(views, self.vision, hint=hint, log=self._log)
            # Register genuinely overlapping views before any generative operation.
            # Limit pair attempts, not source inspection, to keep costs bounded.
            for a, anchor in enumerate(views[:8]):
                for b, donor in enumerate(views[a + 1:8], a + 1):
                    self._check_cancelled()
                    assembled = assemble_overlap(anchor, donor)
                    if assembled is not None:
                        merged, mask, record = assembled
                        folder = self.run_dir / "diagnostics" / f"assembly_{a:02d}_{b:02d}"
                        save_png(merged, folder / "evidence.png")
                        save_png(mask, folder / "observed_mask.png")
                        record["evidence_image"] = str(folder / "evidence.png")
                        self.evidence.assemblies.append((merged, record))
            manifest["warnings"] = self.evidence.warnings
            write_json(self.run_dir / "subject_evidence.json", self.evidence.report())
            for warning in self.evidence.warnings:
                self._log("REVIEW: " + warning)
            for index, part in enumerate(self.evidence.parts):
                save_png(part.image, self.run_dir / "diagnostics" / "parts" / f"{index:02d}_{safe_stem(part.name)}.png")
            subject = SubjectKind.PERSON if subject_key(self.evidence.label) == "person" else SubjectKind.OBJECT
            self.backend = EvidenceSdxlBackend(subject_kind=subject, device=self.config.device,
                models=self.config.model_settings, use_face_adapter=False, log=self._log)
            self.callbacks.progress(0, 0, "Loading reconstruction model...")
            self.backend.prepare()
            tasks = ([None] if self.complete_subject else views) + [None] * self.config.synthetic_variants
            synthetic_index = 0
            for index, view in enumerate(tasks):
                self._check_cancelled()
                label = f"Reconstruction {index + 1}/{len(tasks)}"
                self.callbacks.progress(index, len(tasks), label)
                for semantic_attempt in range(2):
                    try:
                        record = self._reconstruct_view(view, synthetic_index, index, len(tasks), semantic_attempt)
                        break
                    except EvidenceConsistencyError as exc:
                        if semantic_attempt:
                            raise
                        self._log(f"Unsupported generated detail: {exc}. Retrying the view once.")
                if view is None:
                    synthetic_index += 1
                self._record_output(record, result, manifest)
            if not result.output_paths:
                raise BackendFatalError("No subject reconstruction was produced")
            manifest["status"] = "completed_with_errors" if result.errors else "completed"
            self.callbacks.progress(len(tasks), len(tasks), f"Saved {len(result.output_paths)} reconstructed view(s)")
        except (GenerationCancelled, PipelineCancelled):
            result.cancelled = True
            manifest["status"] = "cancelled"
        except Exception as exc:
            # Application boundary: keep failures and completed outputs, then propagate.
            manifest["status"] = "failed"
            self._record_error(exc, result, manifest, "joint reconstruction")
            raise
        finally:
            for resource in (self.vision, self.backend, self._upscaler):
                if resource is not None:
                    try:
                        resource.close()
                    except (OSError, RuntimeError) as exc:
                        self._log(f"Cleanup warning: {exc}")
            manifest["finished_at"] = datetime.now().astimezone().isoformat()
            try:
                write_json(result.manifest_path, manifest)
            finally:
                self.logger.removeHandler(handler)
                handler.close()
        return result

    def _request(self, canvas, missing, references, prompt, negative, seed, *, free=False, target_masks=()):
        return EvidenceRequest(canvas=canvas, generated_mask=missing, reference_board=references[0],
            face_reference_board=None, prompt=prompt, negative_prompt=negative, seed=seed,
            steps=self.config.inference_steps, guidance_scale=self.config.guidance_scale,
            strength=1.0 if free else self.config.denoising_strength,
            reference_fidelity=self.config.reference_fidelity,
            width=canvas.width, height=canvas.height, fully_synthetic=free,
            reference_images=references, free_composition=free, reference_masks=target_masks)

    def _call(self, request, done, total, label):
        self._check_cancelled()
        return self.backend.generate(request, cancel_event=self.cancel_event,
            progress=lambda step, steps, message: self.callbacks.progress(done, total, f"{label}: {step}/{steps}"))

    def _reconstruct_view(self, view, synthetic_index, done, total, semantic_attempt=0):
        references, reference_records = self.evidence.references()
        synthetic = view is None
        seed = self._seed_for(1, done + 1, 1) + 1000 * semantic_attempt
        recovered = []
        prompt, negative = full_subject_prompt(self.evidence, synthetic_index, self.config.custom_prompt)
        if synthetic:
            frame = VariantFrame.FULL_BODY if subject_key(self.evidence.label) == "person" else VariantFrame.SQUARE
            canvas = prepare_variant_canvas(frame=frame, target_long_edge=self.config.target_long_edge)
        else:
            image, observed, recovered = repair_from_overlaps(view, self.evidence.views, self._check_cancelled)
            canvas = prepare_outpaint_canvas(image, observed, aspect=self.config.completion_aspect,
                margin_percent=self.config.completion_margin_percent, target_long_edge=self.config.target_long_edge)
            prompt = (f"Restore missing portions of the same {self.evidence.label}, preserve this photo's "
                      "perspective and its visible details. Use the complementary source parts. " + self.config.custom_prompt)
        framing = None
        for attempt in range(2 if synthetic else 1):
            request = self._request(canvas.image, canvas.generated_mask, references, prompt, negative,
                                    seed + attempt, free=synthetic)
            draft = self._call(request, done, total, "Creating a complete composition" if synthetic else "Restoring source photo")
            draft = preserve_observed_pixels(draft, canvas.image, canvas.generated_mask)
            if not synthetic:
                break
            self.callbacks.progress(done, total, "Checking whole-subject framing...")
            framing = check_whole_framing(self.vision.locate(draft, self.evidence.label), draft.size)
            if framing is None:
                break
            self._log(f"Framing check: {framing}. Attempt {attempt + 1}/2.")
            prompt += " Wide shot, small subject occupying the middle half of the frame, everything visible."
        if synthetic and framing:
            raise BackendFatalError(f"Could not produce a safely framed whole subject after two attempts: {framing}")
        self._check_cancelled()
        slug = f"subject__whole_{synthetic_index + 1:02d}" if synthetic else f"source__{done + 1:03d}_{safe_stem(view.path.stem)}"
        save_png(draft, self.run_dir / "diagnostics" / slug / "draft.png")
        draft_metadata = dict(self.backend.last_generation_metadata)
        final, details = self._refine_parts(draft, canvas.generated_mask, prompt, negative, seed, done, total)
        final = preserve_observed_pixels(final, canvas.image, canvas.generated_mask)
        # Inspect again after regional editing; no failed candidate is exported as final.
        if synthetic:
            framing = check_whole_framing(self.vision.locate(final, self.evidence.label), final.size)
            if framing:
                raise BackendFatalError("Regional reconstruction changed whole-subject framing: " + framing)
        if synthetic and "sunglasses" in negative:
            labels = [r.label for r in self.vision.describe_regions(final)]
            if any(re.search(r"\b(sunglasses|goggles|eyeglasses|spectacles)\b", text) for text in labels):
                raise EvidenceConsistencyError("Eyewear was generated despite visible eye references without it")
        return self._save_output(image=final, source_canvas=canvas.image,
            observed_mask=canvas.observed_mask, generated_mask=canvas.generated_mask,
            output_base=self.run_dir / "images" / slug,
            metadata={"kind": "whole_subject" if synthetic else "joint_source_repair",
                      "fully_synthetic": synthetic, "seed": seed, "semantic_attempt": semantic_attempt + 1, "prompt": prompt, "negative_prompt": negative,
                      "reference_evidence": reference_records, "parts_applied": details,
                      "same_view_recovery": recovered, "draft_generation": draft_metadata,
                      "framing_check": "bounding_box_margin_passed" if synthetic else "source_view_preserved",
                      "identity_accuracy": "not_verified", "complete_geometry": "not_verified",
                      "notice": "New views are generated hypotheses; all local detail passes use original evidence, not generated references."})

    def _refine_parts(self, image, allowed_mask, prompt, negative, seed, done, total):
        planned, records = [], []
        for part in self.evidence.parts[:4]:
            self._check_cancelled()
            self.callbacks.progress(done, total, f"Matching complementary part: {part.name}")
            targets = self.vision.locate(image, part.name)
            if not targets:
                records.append({**part.record(), "status": "not_localized_in_output"})
                continue
            target = max(targets, key=lambda region: region.area)
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle(target.box, fill=255)
            mask = ImageChops.multiply(mask, allowed_mask)
            if mask.getbbox() is None or np.count_nonzero(np.asarray(mask)) < 64:
                records.append({**part.record(), "status": "outside_missing_region_or_too_small"})
                continue
            planned.append((target.area, part, mask, target.box))
        # Broad context first, small eye/mouth/object-detail regions last.
        for index, (_area, part, mask, box) in enumerate(sorted(planned, key=lambda item: item[0], reverse=True)):
            local_prompt = (f"A realistic {part.name} of the same {self.evidence.label}, matching the original "
                            "reference detail, seamlessly integrated with the surrounding photograph. " + prompt)
            request = self._request(image, mask, (part.image,), local_prompt, negative,
                                    seed + 100 + index, target_masks=(mask,))
            # Refine from a valid draft instead of diffusing every detail from noise.
            request.strength = 0.75
            replacement = self._call(request, done, total, f"Reconstructing {part.name} from {part.source.name}")
            # Feather inward only. Never change an observed pixel outside the allowed mask.
            blend = ImageChops.multiply(mask.filter(ImageFilter.GaussianBlur(3)), mask)
            image = Image.composite(replacement, image, blend)
            records.append({**part.record(), "target_box": box, "status": "regionally_conditioned",
                            "pixel_copy": False, "generation": dict(self.backend.last_generation_metadata)})
        return image, records
