"""Joint subject reconstruction: originals -> evidence -> whole views -> local details.

Source repair and free composition have different coordinate systems. Generated
views are never passed off as recovered observations or fed back as source evidence.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from bodybuilder.ai.base import BackendFatalError, GenerationCancelled
from bodybuilder.ai.evidence_sdxl import EvidenceRequest, EvidenceSdxlBackend
from bodybuilder.ai.subject_vision import SubjectVision
from bodybuilder.config import BackendKind, SubjectKind, VariantFrame
from bodybuilder.core.canvas import (
    prepare_outpaint_canvas,
    prepare_variant_canvas,
    preserve_observed_pixels,
)
from bodybuilder.core.fragment_registration import assemble_overlap, repair_from_overlaps
from bodybuilder.core.framing import FramingAssessment, assess_framing, make_recovery_canvas
from bodybuilder.core.image_io import (
    ensure_unique_path,
    load_fragment,
    safe_stem,
    save_png,
    write_json,
)
from bodybuilder.core.pipeline import (
    PipelineCancelled,
    PipelineRunResult,
    ReconstructionPipeline,
    analyze_input_folder,
    environment_report,
)
from bodybuilder.core.run_options import QualityMode, WorkflowMode, plan_tasks, quality_profile
from bodybuilder.core.subject_evidence import (
    SourceView,
    build_evidence,
    full_subject_prompt,
    subject_key,
)


class EvidenceConsistencyError(BackendFatalError):
    """A semantic check found an unsupported change, not a numerical failure."""


@dataclass(slots=True)
class JointRunResult(PipelineRunResult):
    review_paths: list[str] = field(default_factory=list)
    review_messages: list[str] = field(default_factory=list)
    restored_paths: list[str] = field(default_factory=list)
    synthetic_paths: list[str] = field(default_factory=list)


class JointReconstructionPipeline(ReconstructionPipeline):
    def __init__(self, config, *, complete_subject=True, subject_hint="",
                 workflow: WorkflowMode | None = None, quality: QualityMode | None = None, **kwargs):
        self.profile = quality_profile(quality) if quality is not None else None
        if self.profile is not None:
            config = self.profile.apply(config)
        super().__init__(config, **kwargs)
        # Explicit desktop workflows count TOTAL synthetic views. Keep the older
        # complete_subject/extra-variants API working for existing scripts.
        self.workflow = WorkflowMode(workflow) if workflow is not None else None
        self.complete_subject = complete_subject
        self.subject_hint = subject_hint
        self.vision = SubjectVision(self.cancel_event, self._log)
        self.evidence = None
        self.backend = None

    def _build_tasks(self, views):
        if self.workflow is None:
            return ([None] if self.complete_subject else views) + [None] * self.config.synthetic_variants
        return plan_tasks(views, self.workflow, self.config.synthetic_variants,
                          completions_per_source=self.config.completions_per_source)

    def _quality_report(self):
        if self.profile is not None:
            return self.profile.report()
        return {"mode": "custom", "working_long_edge": self.config.target_long_edge,
                "generation_steps": self.config.inference_steps, "detail_steps": self.config.inference_steps,
                "max_detail_passes": 4, "reference_limit": 16}

    def run(self) -> JointRunResult:
        if self.config.backend != BackendKind.SDXL:
            raise ValueError("Joint subject reconstruction requires the real AI backend")
        self._validate_paths()
        name = safe_stem(self.config.run_name) if self.config.run_name else datetime.now().strftime("BodyBuilder_%Y%m%d_%H%M%S")
        self.run_dir = ensure_unique_path(self.config.output_dir / name)
        self.run_dir.mkdir(parents=True)
        self.logger = logging.getLogger(f"bodybuilder.joint.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.run_dir / "bodybuilder.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        result = JointRunResult(self.run_dir, self.run_dir / "run_manifest.json")
        workflow_name = self.workflow.value if self.workflow is not None else (
            "whole_subject" if self.complete_subject else "joint_source_repair")
        manifest = {"status": "running", "workflow": workflow_name,
                    "quality_profile": self._quality_report(),
                    "output_counts": {"restored_sources": 0, "synthetic_views": 0},
                    "config": self.config.to_dict(), "environment": environment_report(),
                    "sources": [], "outputs": [], "errors": [], "warnings": [],
                    "review_required": False, "review_count": 0}
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
            tasks = self._build_tasks(views)
            manifest["planned_outputs"] = {
                "restored_sources": sum(view is not None for view in tasks),
                "synthetic_views": sum(view is None for view in tasks),
            }
            self._log(f"Planned outputs: {manifest['planned_outputs']}. Quality: {self._quality_report()}")
            write_json(result.manifest_path, manifest)
            hint = self.subject_hint or ("person" if self.config.subject_kind == SubjectKind.PERSON else "")
            self.evidence = build_evidence(views, self.vision, hint=hint, log=self._log)
            # Register genuine overlaps before generation, with a bounded pair budget.
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
            manifest["warnings"] = list(self.evidence.warnings)
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
            # The evidence and model are shared by both output types. Generated
            # source restorations are never appended to the original references.
            synthetic_index = 0
            for index, view in enumerate(tasks):
                self._check_cancelled()
                task_name = (f"Synthetic view {synthetic_index + 1}" if view is None
                             else f"Restore source {view.path.name}")
                label = f"{task_name} ({index + 1}/{len(tasks)})"
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
                if record.get("review_required"):
                    result.review_paths.append(record["image"])
                    for warning in record["warnings"]:
                        message = f"{record['image']}: {warning}"
                        result.review_messages.append(message)
                        manifest["warnings"].append(message)
                    manifest["review_required"] = True
                    manifest["review_count"] = len(result.review_paths)
                if view is None:
                    result.synthetic_paths.append(record["image"])
                else:
                    result.restored_paths.append(record["image"])
                manifest["output_counts"] = {"restored_sources": len(result.restored_paths),
                                             "synthetic_views": len(result.synthetic_paths)}
                self._record_output(record, result, manifest)
            if not result.output_paths:
                raise BackendFatalError("No subject reconstruction was produced")
            if result.errors:
                manifest["status"] = "completed_with_errors"
            elif result.review_paths:
                manifest["status"] = "completed_with_warnings"
            else:
                manifest["status"] = "completed"
            self.callbacks.progress(len(tasks), len(tasks),
                f"Saved {len(result.output_paths)} view(s); {len(result.review_paths)} need framing review")
        except (GenerationCancelled, PipelineCancelled):
            result.cancelled = True
            manifest["status"] = "cancelled"
        except Exception as exc:
            # Operational/model errors still fail explicitly; only framing is advisory.
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

    def _assess_frame(self, image: Image.Image, done: int, total: int) -> FramingAssessment:
        self._check_cancelled()
        self.callbacks.progress(done, total, "Checking framing (advisory, not proof of completeness)...")
        regions = self.vision.locate(image, self.evidence.label)
        self._check_cancelled()
        return assess_framing((region.box for region in regions), image.size)

    def _reconstruct_view(self, view, synthetic_index, done, total, semantic_attempt=0):
        limit = self.profile.reference_limit if self.profile is not None else 16
        references, reference_records = self.evidence.references(limit=limit)
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
        slug = f"subject__whole_{synthetic_index + 1:02d}" if synthetic else f"source__{done + 1:03d}_{safe_stem(view.path.stem)}"
        diagnostics = self.run_dir / "diagnostics" / slug / f"semantic_{semantic_attempt + 1:02d}"
        request = self._request(canvas.image, canvas.generated_mask, references, prompt, negative, seed, free=synthetic)
        draft = self._call(request, done, total, "Creating a complete composition" if synthetic else "Restoring source photo")
        draft = preserve_observed_pixels(draft, canvas.image, canvas.generated_mask)
        draft_metadata = dict(self.backend.last_generation_metadata)
        save_png(draft, diagnostics / "draft_01.png")
        framing_history = []
        selected_draft = 1
        if synthetic:
            assessment = self._assess_frame(draft, done, total)
            framing_history.append({"stage": "initial", "seed": seed, **assessment.report(),
                                    "generation": draft_metadata})
            write_json(diagnostics / "framing_history.json", framing_history)
            if not assessment.passed:
                self._log(f"Framing review: {assessment.message}. Trying one bounded layout correction.")
                correction_prompt = prompt + " Wide shot, one small centered subject, entire silhouette visible with space on every side."
                if assessment.status == "edge_contact":
                    # A real inpainting pass completes the new margins. The central
                    # generated draft is protected, but never counted as source evidence.
                    recovery = make_recovery_canvas(draft)
                    retry_canvas, retry_mask = recovery.image, recovery.missing
                    correction_prompt += " Continue the subject and background naturally into the missing margins."
                    correction_kind = "outpaint_generated_draft"
                    placement = recovery.draft_box
                    free = False
                    save_png(retry_canvas, diagnostics / "recovery_canvas.png")
                    save_png(retry_mask, diagnostics / "recovery_generated_mask.png")
                else:
                    retry_canvas, retry_mask = canvas.image, canvas.generated_mask
                    correction_kind, placement, free = "new_composition", None, True
                retry = self._request(retry_canvas, retry_mask, references,
                    correction_prompt, negative, seed + 1, free=free)
                retry.strength = 1.0
                candidate = self._call(retry, done, total, "Correcting subject framing")
                candidate = preserve_observed_pixels(candidate, retry_canvas, retry_mask)
                candidate_metadata = dict(self.backend.last_generation_metadata)
                save_png(candidate, diagnostics / "draft_02.png")
                corrected = self._assess_frame(candidate, done, total)
                framing_history.append({"stage": correction_kind, "seed": seed + 1,
                    "prompt": correction_prompt, "draft_box": placement,
                    "draft_is_generated_not_observed": True, **corrected.report(),
                    "generation": candidate_metadata})
                write_json(diagnostics / "framing_history.json", framing_history)
                if corrected.rank <= assessment.rank:
                    draft, draft_metadata = candidate, candidate_metadata
                    selected_draft = 2
                else:
                    self._log("Layout correction had a weaker detector assessment; retaining the initial draft.")
        self._check_cancelled()
        final, details = self._refine_parts(draft, canvas.generated_mask, prompt, negative, seed, done, total)
        final = preserve_observed_pixels(final, canvas.image, canvas.generated_mask)
        warnings = []
        final_check = None
        if synthetic:
            final_check = self._assess_frame(final, done, total)
            framing_history.append({"stage": "after_regional_refinement", **final_check.report()})
            write_json(diagnostics / "framing_history.json", framing_history)
            if not final_check.passed:
                warnings.append("Framing needs visual review: " + final_check.message +
                    ". Image retained; complete anatomy or geometry has not been confirmed.")
                self._log("REVIEW: " + warnings[-1])
        # Keep unrelated consistency/model errors distinct from uncertain framing.
        if synthetic and "sunglasses" in negative:
            labels = [r.label for r in self.vision.describe_regions(final)]
            if any(re.search(r"\b(sunglasses|goggles|eyeglasses|spectacles)\b", text) for text in labels):
                raise EvidenceConsistencyError("Eyewear was generated despite visible eye references without it")
        output_slug = slug + "__needs_review" if warnings else slug
        if self.config.upscale_2x:
            self.callbacks.progress(done, total, "Enhancing output 2x while protecting observed details...")
        return self._save_output(image=final, source_canvas=canvas.image,
            observed_mask=canvas.observed_mask, generated_mask=canvas.generated_mask,
            output_base=self.run_dir / "images" / output_slug,
            metadata={"kind": "whole_subject" if synthetic else "joint_source_repair",
                      "source_file": str(view.path) if view is not None else None,
                      "quality_profile": self._quality_report(),
                      "fully_synthetic": synthetic, "seed": seed, "semantic_attempt": semantic_attempt + 1,
                      "prompt": prompt, "negative_prompt": negative,
                      "reference_evidence": reference_records, "parts_applied": details,
                      "same_view_recovery": recovered, "draft_generation": draft_metadata,
                      "selected_draft": selected_draft, "framing_history": framing_history,
                      "framing_check": ("needs_review" if warnings else "bounding_box_margin_passed") if synthetic else "source_view_preserved",
                      "review_required": bool(warnings), "warnings": warnings,
                      "identity_accuracy": "not_verified", "complete_geometry": "not_verified",
                      "notice": "New views are generated hypotheses; original photographs alone supply the references. Framing checks are advisory."})

    def _refine_parts(self, image, allowed_mask, prompt, negative, seed, done, total):
        planned, records = [], []
        max_passes = self.profile.max_detail_passes if self.profile is not None else 4
        for part in self.evidence.parts[:max_passes]:
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
        for index, (_area, part, mask, box) in enumerate(sorted(planned, key=lambda item: item[0], reverse=True)):
            local_prompt = (f"A realistic {part.name} of the same {self.evidence.label}, matching the original "
                            "reference detail, seamlessly integrated with the surrounding photograph. " + prompt)
            request = self._request(image, mask, (part.image,), local_prompt, negative,
                                    seed + 100 + index, target_masks=(mask,))
            request.strength = 0.75
            if self.profile is not None:
                request.steps = self.profile.detail_steps
            replacement = self._call(request, done, total, f"Reconstructing {part.name} from {part.source.name}")
            blend = ImageChops.multiply(mask.filter(ImageFilter.GaussianBlur(3)), mask)
            image = Image.composite(replacement, image, blend)
            records.append({**part.record(), "target_box": box, "status": "regionally_conditioned",
                            "pixel_copy": False, "generation": dict(self.backend.last_generation_metadata)})
        return image, records
