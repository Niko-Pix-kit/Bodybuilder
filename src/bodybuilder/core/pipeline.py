"""End-to-end reconstruction pipeline with cancellation and provenance."""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from bodybuilder import __version__
from bodybuilder.ai.base import (
    BackendFatalError,
    ClassicalFillBackend,
    GenerationBackend,
    GenerationCancelled,
)
from bodybuilder.ai.sdxl import SdxlBackend
from bodybuilder.ai.upscale import LanczosUpscaler, Swin2SRUpscaler, Upscaler
from bodybuilder.config import BackendKind, PipelineConfig, SubjectKind
from bodybuilder.core.analysis import analyze_paths, best_face_crops, make_reference_board
from bodybuilder.core.canvas import (
    generated_fraction,
    prepare_outpaint_canvas,
    prepare_variant_canvas,
    preserve_observed_pixels,
)
from bodybuilder.core.clustering import group_images
from bodybuilder.core.image_io import (
    ensure_unique_path,
    load_rgb,
    safe_stem,
    save_png,
    scan_images,
    write_json,
)
from bodybuilder.core.prompts import completion_prompt, variant_prompt
from bodybuilder.core.provenance import create_comparison, save_jpeg
from bodybuilder.core.stitching import stitch_from_anchor
from bodybuilder.core.types import GenerationRequest, ImageAnalysis, StitchResult
from bodybuilder.logging_utils import create_run_logger


class PipelineCancelled(RuntimeError):
    """Raised when the user cancels the active run."""


@dataclass(slots=True)
class PipelineCallbacks:
    log: Callable[[str], None] = lambda _message: None
    progress: Callable[[int, int, str], None] = lambda _current, _total, _message: None
    preview: Callable[[Path], None] = lambda _path: None
    analysis_item: Callable[[ImageAnalysis], None] = lambda _item: None


@dataclass(slots=True)
class AnalysisRunResult:
    analyses: list[ImageAnalysis]
    failures: list[dict[str, str]]


@dataclass(slots=True)
class PipelineRunResult:
    run_dir: Path
    manifest_path: Path
    output_paths: list[Path] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False


def analyze_input_folder(
    config: PipelineConfig,
    *,
    callbacks: PipelineCallbacks | None = None,
    cancel_event: threading.Event | None = None,
) -> AnalysisRunResult:
    callbacks = callbacks or PipelineCallbacks()
    cancel_event = cancel_event or threading.Event()
    paths = scan_images(config.input_dir, recursive=config.recursive_scan)
    if not paths:
        raise RuntimeError("No supported image files were found in the selected input folder")

    analyses, failures = analyze_paths(
        paths,
        progress=callbacks.progress,
        cancelled=cancel_event.is_set,
    )
    for item in analyses:
        callbacks.analysis_item(item)
    for failure in failures:
        callbacks.log(
            f"Could not decode {failure.get('path', 'unknown file')}: "
            f"{failure.get('error', 'unknown error')}"
        )
    if cancel_event.is_set():
        raise PipelineCancelled("Analysis cancelled")
    if not analyses:
        raise RuntimeError("Every discovered image failed to decode")
    return AnalysisRunResult(analyses=analyses, failures=failures)


class ReconstructionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        callbacks: PipelineCallbacks | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.callbacks = callbacks or PipelineCallbacks()
        self.cancel_event = cancel_event or threading.Event()
        self.run_dir: Path | None = None
        self.logger: logging.Logger | None = None
        self._backends: dict[tuple[SubjectKind, bool], GenerationBackend] = {}
        self._upscaler: Upscaler | None = None
        self._upscaler_failed = False

    def run(self) -> PipelineRunResult:
        self._validate_paths()
        self.run_dir = self._create_run_directory()
        self.logger = create_run_logger(self.run_dir / "bodybuilder.log")
        started_at = datetime.now().astimezone()
        manifest_path = self.run_dir / "run_manifest.json"
        result = PipelineRunResult(run_dir=self.run_dir, manifest_path=manifest_path)
        manifest: dict[str, Any] = {
            "application": "BodyBuilder",
            "version": __version__,
            "started_at": started_at.isoformat(),
            "status": "running",
            "truth_notice": (
                "Missing pixels are generated hypotheses. Source completions preserve observed pixels; "
                "synthetic variants are entirely generated."
            ),
            "config": self.config.to_dict(),
            "subjects": [],
            "outputs": [],
            "warnings": [],
            "errors": [],
        }
        write_json(manifest_path, manifest)
        self._log(f"Run directory: {self.run_dir}")

        try:
            paths = scan_images(self.config.input_dir, recursive=self.config.recursive_scan)
            if not paths:
                raise RuntimeError("No supported image files were found in the selected input folder")
            self._log(f"Discovered {len(paths)} supported image file(s).")

            analyses, analysis_failures = analyze_paths(
                paths,
                progress=self.callbacks.progress,
                cancelled=self.cancel_event.is_set,
            )
            for analysis in analyses:
                self.callbacks.analysis_item(analysis)
            for failure in analysis_failures:
                error = {"stage": "analysis", **failure}
                manifest["errors"].append(error)
                result.errors.append(error)
            self._check_cancelled()
            if not analyses:
                raise RuntimeError("Every discovered image failed to decode")

            models = self.config.model_settings
            grouping = group_images(
                analyses,
                group_all=self.config.group_all_images,
                threshold=self.config.grouping_similarity_threshold,
                model_id=models.grouping_model_id,
                device=self.config.device,
                log=self._log,
            )
            if grouping.warning:
                manifest["warnings"].append(grouping.warning)

            variant_count = self.config.synthetic_variants
            if self.config.backend == BackendKind.CLASSICAL and variant_count:
                warning = (
                    "Synthetic variants were skipped because the classical backend cannot create "
                    "new poses or views."
                )
                self._log(warning)
                manifest["warnings"].append(warning)
                variant_count = 0

            total_generation_tasks = sum(
                len(group) * self.config.completions_per_source + variant_count
                for group in grouping.groups
            )
            completed_generation_tasks = 0

            for subject_index, group in enumerate(grouping.groups, start=1):
                self._check_cancelled()
                subject_folder = self.run_dir / f"subject_{subject_index:03d}"
                subject_folder.mkdir(parents=True, exist_ok=True)
                subject_kind = self._resolve_subject_kind(group)
                self._log(
                    f"Subject {subject_index}: {len(group)} image(s), resolved type "
                    f"{subject_kind.value}."
                )

                sorted_group = sorted(group, key=lambda item: item.quality_score, reverse=True)
                reference_images = [load_rgb(item.path) for item in sorted_group[:16]]
                reference_board = make_reference_board(
                    reference_images,
                    title="All source references",
                )
                save_jpeg(reference_board, subject_folder / "reference_board.jpg", quality=94)

                face_crops = best_face_crops(sorted_group)
                face_board: Image.Image | None = None
                if face_crops:
                    face_board = make_reference_board(
                        face_crops,
                        max_images=8,
                        title="Detected face references",
                    )
                    save_jpeg(face_board, subject_folder / "face_reference_board.jpg", quality=94)

                subject_manifest: dict[str, Any] = {
                    "subject_index": subject_index,
                    "resolved_kind": subject_kind.value,
                    "source_count": len(group),
                    "face_reference_count": len(face_crops),
                    "sources": [item.to_dict() for item in group],
                    "outputs": [],
                }
                manifest["subjects"].append(subject_manifest)
                write_json(subject_folder / "analysis.json", subject_manifest)
                write_json(manifest_path, manifest)

                backend = self._backend_for(
                    subject_kind,
                    use_face_adapter=face_board is not None,
                )
                backend.prepare()
                diagnostics_dir = subject_folder / "diagnostics"
                completions_dir = subject_folder / "completions"
                variants_dir = subject_folder / "variants"

                for source_index, analysis in enumerate(group, start=1):
                    self._check_cancelled()
                    source_slug = f"source_{source_index:03d}_{safe_stem(analysis.path.stem)}"
                    try:
                        stitched = self._prepare_observed_source(analysis, group)
                        canvas = prepare_outpaint_canvas(
                            stitched.image,
                            stitched.observed_mask,
                            aspect=self.config.completion_aspect,
                            margin_percent=self.config.completion_margin_percent,
                            target_long_edge=self.config.target_long_edge,
                        )
                        save_png(canvas.image, diagnostics_dir / f"{source_slug}_canvas.png")
                        save_png(
                            canvas.observed_mask,
                            diagnostics_dir / f"{source_slug}_observed_mask.png",
                        )
                        save_png(
                            canvas.generated_mask,
                            diagnostics_dir / f"{source_slug}_generated_mask.png",
                        )

                        prompt, negative_prompt = completion_prompt(
                            subject_kind,
                            self.config.custom_prompt,
                        )
                        for completion_index in range(1, self.config.completions_per_source + 1):
                            self._check_cancelled()
                            seed = self._seed_for(subject_index, source_index, completion_index)
                            task_label = (
                                f"Subject {subject_index}, source {source_index}, "
                                f"completion {completion_index}"
                            )
                            self.callbacks.progress(
                                completed_generation_tasks,
                                max(1, total_generation_tasks),
                                task_label,
                            )
                            request = GenerationRequest(
                                canvas=canvas.image,
                                generated_mask=canvas.generated_mask,
                                reference_board=reference_board,
                                face_reference_board=face_board,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                seed=seed,
                                steps=self.config.inference_steps,
                                guidance_scale=self.config.guidance_scale,
                                strength=self.config.denoising_strength,
                                reference_fidelity=self.config.reference_fidelity,
                                width=canvas.image.width,
                                height=canvas.image.height,
                                fully_synthetic=False,
                            )
                            generated = backend.generate(
                                request,
                                cancel_event=self.cancel_event,
                                progress=lambda step, steps, message, current=completed_generation_tasks, label=task_label: self.callbacks.progress(
                                    current,
                                    max(1, total_generation_tasks),
                                    f"{label}: {message} {step}/{steps}",
                                ),
                            )
                            reconstructed = preserve_observed_pixels(
                                generated,
                                canvas.image,
                                canvas.generated_mask,
                            )
                            output_base = (
                                completions_dir
                                / f"{source_slug}__completion_{completion_index:02d}"
                            )
                            output_record = self._save_output(
                                image=reconstructed,
                                source_canvas=canvas.image,
                                observed_mask=canvas.observed_mask,
                                generated_mask=canvas.generated_mask,
                                output_base=output_base,
                                metadata={
                                    "kind": "source_completion",
                                    "subject_index": subject_index,
                                    "source_index": source_index,
                                    "source_file": str(analysis.path),
                                    "source_files_used": [str(path) for path in stitched.used_paths],
                                    "source_files_rejected_for_stitching": [
                                        str(path) for path in stitched.rejected_paths
                                    ],
                                    "seed": seed,
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt,
                                    "backend": backend.name,
                                    "model": backend.model_identifier,
                                    "observed_working_pixels_preserved_after_diffusion": True,
                                    "source_resampled_for_ai_canvas": stitched.image.size
                                    != (canvas.placement.source_width, canvas.placement.source_height),
                                    "observed_input_width": stitched.image.width,
                                    "observed_input_height": stitched.image.height,
                                    "fully_synthetic": False,
                                    "placement": canvas.placement.to_dict(),
                                    "generated_fraction": round(
                                        generated_fraction(canvas.generated_mask), 6
                                    ),
                                },
                            )
                            completed_generation_tasks += 1
                            result.output_paths.append(Path(output_record["image"]))
                            manifest["outputs"].append(output_record)
                            subject_manifest["outputs"].append(output_record)
                            write_json(subject_folder / "analysis.json", subject_manifest)
                            write_json(manifest_path, manifest)
                    except (BackendFatalError, GenerationCancelled, PipelineCancelled):
                        raise
                    except Exception as exc:
                        error = self._record_error(
                            manifest,
                            result,
                            stage="source_completion",
                            subject_index=subject_index,
                            source=str(analysis.path),
                            exc=exc,
                        )
                        subject_manifest.setdefault("errors", []).append(error)
                        write_json(subject_folder / "analysis.json", subject_manifest)
                        write_json(manifest_path, manifest)
                        if not self.config.continue_on_error:
                            raise

                for variant_index in range(variant_count):
                    self._check_cancelled()
                    try:
                        slug, prompt, negative_prompt = variant_prompt(
                            subject_kind,
                            variant_index,
                            self.config.variant_frame,
                            self.config.custom_prompt,
                        )
                        variant_canvas = prepare_variant_canvas(
                            frame=self.config.variant_frame,
                            target_long_edge=self.config.target_long_edge,
                        )
                        seed = self._seed_for(subject_index, 10_000, variant_index + 1)
                        task_label = (
                            f"Subject {subject_index}, synthetic variant {variant_index + 1}"
                        )
                        self.callbacks.progress(
                            completed_generation_tasks,
                            max(1, total_generation_tasks),
                            task_label,
                        )
                        request = GenerationRequest(
                            canvas=variant_canvas.image,
                            generated_mask=variant_canvas.generated_mask,
                            reference_board=reference_board,
                            face_reference_board=face_board,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            seed=seed,
                            steps=self.config.inference_steps,
                            guidance_scale=self.config.guidance_scale,
                            strength=1.0,
                            reference_fidelity=self.config.reference_fidelity,
                            width=variant_canvas.image.width,
                            height=variant_canvas.image.height,
                            fully_synthetic=True,
                        )
                        generated = backend.generate(
                            request,
                            cancel_event=self.cancel_event,
                            progress=lambda step, steps, message, current=completed_generation_tasks, label=task_label: self.callbacks.progress(
                                current,
                                max(1, total_generation_tasks),
                                f"{label}: {message} {step}/{steps}",
                            ),
                        )
                        output_base = variants_dir / f"variant_{variant_index + 1:03d}_{slug}"
                        output_record = self._save_output(
                            image=generated,
                            source_canvas=variant_canvas.image,
                            observed_mask=variant_canvas.observed_mask,
                            generated_mask=variant_canvas.generated_mask,
                            output_base=output_base,
                            metadata={
                                "kind": "synthetic_variant",
                                "subject_index": subject_index,
                                "seed": seed,
                                "prompt": prompt,
                                "negative_prompt": negative_prompt,
                                "backend": backend.name,
                                "model": backend.model_identifier,
                                "observed_pixels_preserved": False,
                                "fully_synthetic": True,
                                "generated_fraction": 1.0,
                            },
                        )
                        completed_generation_tasks += 1
                        result.output_paths.append(Path(output_record["image"]))
                        manifest["outputs"].append(output_record)
                        subject_manifest["outputs"].append(output_record)
                        write_json(subject_folder / "analysis.json", subject_manifest)
                        write_json(manifest_path, manifest)
                    except (BackendFatalError, GenerationCancelled, PipelineCancelled):
                        raise
                    except Exception as exc:
                        error = self._record_error(
                            manifest,
                            result,
                            stage="synthetic_variant",
                            subject_index=subject_index,
                            variant_index=variant_index + 1,
                            exc=exc,
                        )
                        subject_manifest.setdefault("errors", []).append(error)
                        write_json(subject_folder / "analysis.json", subject_manifest)
                        write_json(manifest_path, manifest)
                        if not self.config.continue_on_error:
                            raise

            manifest["status"] = "completed_with_errors" if manifest["errors"] else "completed"
            manifest["completed_at"] = datetime.now().astimezone().isoformat()
            self.callbacks.progress(
                max(1, total_generation_tasks),
                max(1, total_generation_tasks),
                "Reconstruction complete",
            )
        except (PipelineCancelled, GenerationCancelled):
            result.cancelled = True
            manifest["status"] = "cancelled"
            manifest["completed_at"] = datetime.now().astimezone().isoformat()
            self._log("Run cancelled by the user.", level=logging.WARNING)
        except Exception as exc:
            self._record_error(manifest, result, stage="run", exc=exc)
            manifest["status"] = "failed"
            manifest["completed_at"] = datetime.now().astimezone().isoformat()
            write_json(manifest_path, manifest)
            raise
        finally:
            for backend in self._backends.values():
                backend.close()
            if self._upscaler is not None:
                self._upscaler.close()
            write_json(manifest_path, manifest)
        return result

    def _prepare_observed_source(
        self,
        analysis: ImageAnalysis,
        group: list[ImageAnalysis],
    ) -> StitchResult:
        if not self.config.stitch_overlaps or self.config.max_stitch_candidates == 0:
            image = load_rgb(analysis.path)
            return StitchResult(
                image=image,
                observed_mask=Image.new("L", image.size, 255),
                used_paths=[analysis.path],
            )
        candidates = sorted(
            (item for item in group if item.path != analysis.path),
            key=lambda item: item.quality_score,
            reverse=True,
        )
        return stitch_from_anchor(
            analysis.path,
            [item.path for item in candidates],
            max_candidates=self.config.max_stitch_candidates,
            log=self._log,
        )

    def _save_output(
        self,
        *,
        image: Image.Image,
        source_canvas: Image.Image,
        observed_mask: Image.Image,
        generated_mask: Image.Image,
        output_base: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        final_image = image.convert("RGB")
        final_source = source_canvas.convert("RGB")
        final_observed = observed_mask.convert("L")
        final_generated = generated_mask.convert("L")
        enhancement = "none"
        if self.config.upscale_2x:
            upscaler = self._get_upscaler()
            try:
                final_image = upscaler.upscale(final_image)
                final_source = final_source.resize(final_image.size, Image.Resampling.LANCZOS)
                final_observed = final_observed.resize(final_image.size, Image.Resampling.NEAREST)
                final_generated = final_generated.resize(final_image.size, Image.Resampling.NEAREST)
                enhancement = upscaler.name
            except Exception as exc:
                self._upscaler_failed = True
                warning = f"Swin2SR enhancement failed; using Lanczos fallback. Reason: {exc}"
                self._log(warning, level=logging.WARNING)
                fallback = LanczosUpscaler()
                final_image = fallback.upscale(image)
                final_source = source_canvas.resize(final_image.size, Image.Resampling.LANCZOS)
                final_observed = observed_mask.resize(final_image.size, Image.Resampling.NEAREST)
                final_generated = generated_mask.resize(final_image.size, Image.Resampling.NEAREST)
                enhancement = fallback.name

        has_observed_content = final_observed.getbbox() is not None
        if has_observed_content:
            final_image.paste(final_source, (0, 0), final_observed)

        metadata = {
            **metadata,
            "enhancement": enhancement,
            "observed_pixels_preserved": has_observed_content,
            "observed_content_preserved": has_observed_content,
            "observed_working_pixels_preserved_exactly": (
                has_observed_content and enhancement == "none"
            ),
            "observed_region_processing": (
                "copied exactly from the AI working canvas"
                if enhancement == "none"
                else "2x Lanczos resampling from the working canvas; no generative enhancement applied"
            ),
            "output_width": final_image.width,
            "output_height": final_image.height,
            "provenance_notice": "White pixels in generated_mask were synthesized or filled.",
        }
        image_path = output_base.with_suffix(".png")
        observed_path = output_base.with_name(f"{output_base.name}__observed_mask.png")
        generated_path = output_base.with_name(f"{output_base.name}__generated_mask.png")
        comparison_path = output_base.with_name(f"{output_base.name}__comparison.jpg")
        metadata_path = output_base.with_suffix(".json")

        save_png(final_image, image_path, metadata)
        save_png(final_observed, observed_path)
        save_png(final_generated, generated_path)
        comparison = create_comparison(final_source, final_generated, final_image)
        save_jpeg(comparison, comparison_path, quality=92)

        record = {
            **metadata,
            "image": str(image_path),
            "observed_mask": str(observed_path),
            "generated_mask": str(generated_path),
            "comparison": str(comparison_path),
            "metadata": str(metadata_path),
        }
        write_json(metadata_path, record)
        self.callbacks.preview(image_path)
        self._log(f"Saved {image_path}")
        return record

    def _get_upscaler(self) -> Upscaler:
        if self._upscaler_failed:
            return LanczosUpscaler()
        if self._upscaler is None:
            self._upscaler = Swin2SRUpscaler(
                model_id=self.config.model_settings.upscaler_model_id,
                device=self.config.device,
                log=self._log,
            )
        return self._upscaler

    def _backend_for(
        self,
        subject_kind: SubjectKind,
        *,
        use_face_adapter: bool,
    ) -> GenerationBackend:
        key = (subject_kind, use_face_adapter)
        backend = self._backends.get(key)
        if backend is not None:
            return backend

        if self._backends:
            self._log("Releasing the previous generation backend before the next subject group.")
            for previous in self._backends.values():
                previous.close()
            self._backends.clear()

        if self.config.backend == BackendKind.CLASSICAL:
            backend = ClassicalFillBackend()
        else:
            backend = SdxlBackend(
                subject_kind=subject_kind,
                device=self.config.device,
                models=self.config.model_settings,
                use_face_adapter=use_face_adapter,
                log=self._log,
            )
        self._backends[key] = backend
        return backend

    def _resolve_subject_kind(self, group: list[ImageAnalysis]) -> SubjectKind:
        if self.config.subject_kind != SubjectKind.AUTO:
            return self.config.subject_kind
        return SubjectKind.PERSON if any(item.faces for item in group) else SubjectKind.OBJECT

    def _seed_for(self, subject_index: int, source_index: int, variant_index: int) -> int:
        return int(
            (
                self.config.seed
                + subject_index * 1_000_003
                + source_index * 10_007
                + variant_index * 101
            )
            % (2**32 - 1)
        )

    def _record_error(
        self,
        manifest: dict[str, Any],
        result: PipelineRunResult,
        *,
        stage: str,
        exc: Exception,
        **context: Any,
    ) -> dict[str, Any]:
        error = {
            "stage": stage,
            **context,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["errors"].append(error)
        result.errors.append(error)
        if self.logger:
            self.logger.error("Pipeline error: %s\n%s", error, traceback.format_exc())
        self.callbacks.log(f"ERROR [{stage}] {type(exc).__name__}: {exc}")
        return error

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelled("Run cancelled")

    def _validate_paths(self) -> None:
        if not self.config.input_dir.exists() or not self.config.input_dir.is_dir():
            raise RuntimeError(f"Input folder does not exist: {self.config.input_dir}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.config.output_dir.resolve().relative_to(self.config.input_dir.resolve())
        except ValueError:
            return
        raise RuntimeError("The output folder cannot be inside the input folder")

    def _create_run_directory(self) -> Path:
        if self.config.run_name:
            name = safe_stem(self.config.run_name, fallback="BodyBuilder_run")
        else:
            name = datetime.now().astimezone().strftime("BodyBuilder_%Y%m%d_%H%M%S")
        run_dir = ensure_unique_path(self.config.output_dir / name)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _log(self, message: str, *, level: int = logging.INFO) -> None:
        if self.logger:
            self.logger.log(level, message)
        self.callbacks.log(message)
