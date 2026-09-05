"""Reference-guided reconstruction with separate results and diagnostics."""

from __future__ import annotations

import hashlib
import logging
import platform
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
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
from bodybuilder.config import BackendKind, PipelineConfig, SubjectKind
from bodybuilder.core.canvas import (
    PreparedCanvas,
    generated_fraction,
    prepare_outpaint_canvas,
    prepare_variant_canvas,
    preserve_observed_pixels,
)
from bodybuilder.core.image_io import (
    ensure_unique_path,
    load_fragment,
    safe_stem,
    save_png,
    scan_images,
    write_json,
)
from bodybuilder.core.types import GenerationRequest, ImageAnalysis, StitchResult
from bodybuilder.core.validation import mask_like, validate_generation


class PipelineCancelled(RuntimeError):
    """Cancellation at an application checkpoint."""


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


def analyze_input_folder(config: PipelineConfig, *, callbacks: PipelineCallbacks | None = None,
                         cancel_event: threading.Event | None = None) -> AnalysisRunResult:
    from bodybuilder.core.analysis import analyze_paths

    callbacks = callbacks or PipelineCallbacks()
    cancel_event = cancel_event or threading.Event()
    paths = scan_images(config.input_dir, recursive=config.recursive_scan)
    if not paths:
        raise ValueError("No source photographs found. Mask files and previous output folders are excluded.")
    analyses, failures = analyze_paths(paths, progress=callbacks.progress, cancelled=cancel_event.is_set)
    usable = []
    for analysis in analyses:
        if cancel_event.is_set():
            raise PipelineCancelled("Analysis cancelled")
        image, _mask = load_fragment(analysis.path)
        if mask_like(image):
            failures.append({"path": str(analysis.path), "error": "Black/white mask or empty image, not a photograph"})
        else:
            usable.append(analysis)
            callbacks.analysis_item(analysis)
    for failure in failures:
        callbacks.log(f"Skipped {failure['path']}: {failure['error']}")
    if cancel_event.is_set():
        raise PipelineCancelled("Analysis cancelled")
    if not usable:
        raise ValueError("No usable photographs remain. Select the original fragments, not the diagnostic masks.")
    return AnalysisRunResult(usable, failures)


def environment_report() -> dict[str, Any]:
    packages = {}
    for package in ("PyQt6", "Pillow", "numpy", "torch", "diffusers", "transformers", "accelerate"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages}


class ReconstructionPipeline:
    def __init__(self, config: PipelineConfig, *, callbacks: PipelineCallbacks | None = None,
                 cancel_event: threading.Event | None = None) -> None:
        self.config = config
        self.callbacks = callbacks or PipelineCallbacks()
        self.cancel_event = cancel_event or threading.Event()
        self.run_dir: Path | None = None
        self.logger: logging.Logger | None = None
        self._backends: dict[tuple[SubjectKind, bool], GenerationBackend] = {}
        self._upscaler: Any = None

    def run(self) -> PipelineRunResult:
        self._validate_paths()
        name = safe_stem(self.config.run_name) if self.config.run_name else datetime.now().strftime("BodyBuilder_%Y%m%d_%H%M%S")
        self.run_dir = ensure_unique_path(self.config.output_dir / name)
        self.run_dir.mkdir(parents=True)
        self.logger = logging.getLogger(f"bodybuilder.run.{self.run_dir.name}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.run_dir / "bodybuilder.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        result = PipelineRunResult(self.run_dir, self.run_dir / "run_manifest.json")
        manifest: dict[str, Any] = {
            "version": __version__, "status": "running", "config": self.config.to_dict(),
            "environment": environment_report(), "sources": [], "outputs": [], "errors": [],
            "notice": "Missing regions are generated estimates, not recovered evidence.",
        }
        try:
            write_json(result.manifest_path, manifest)
            self._log(f"Run folder: {self.run_dir}")
            analysis_result = analyze_input_folder(self.config, callbacks=self.callbacks, cancel_event=self.cancel_event)
            result.errors.extend(analysis_result.failures)
            manifest["errors"].extend(analysis_result.failures)
            for item in analysis_result.analyses:
                with item.path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                manifest["sources"].append({**item.to_dict(), "sha256": digest})
            groups = self._groups(analysis_result.analyses)
            variants = self.config.synthetic_variants if self.config.backend == BackendKind.SDXL else 0
            total = sum(len(group) * self.config.completions_per_source + variants for group in groups)
            done = 0
            for group_index, group in enumerate(groups, 1):
                self._check_cancelled()
                subject = self._resolve_subject_kind(group)
                backend = self._backend_for(subject, use_face_adapter=False)
                self.callbacks.progress(0, 0, "Loading AI and reference model...")
                backend.prepare()
                self._check_cancelled()
                ordered = sorted(group, key=lambda item: item.quality_score, reverse=True)
                for source_index, analysis in enumerate(group, 1):
                    self._check_cancelled()
                    try:
                        observed = self._prepare_observed_source(analysis, group)
                        canvas = prepare_outpaint_canvas(observed.image, observed.observed_mask,
                            aspect=self.config.completion_aspect,
                            margin_percent=self.config.completion_margin_percent,
                            target_long_edge=self.config.target_long_edge)
                        references, reference_paths = self._references(analysis, ordered)
                        prompt, negative = self._prompts(subject, False, 0)
                        for index in range(1, self.config.completions_per_source + 1):
                            label = f"Photo {source_index}/{len(group)}"
                            record = self._generate(backend, canvas, references, prompt, negative,
                                f"subject_{group_index:03d}__{source_index:03d}_{safe_stem(analysis.path.stem)}__{index:02d}",
                                self._seed_for(group_index, source_index, index), done, total, label,
                                {"kind": "source_completion", "source_file": str(analysis.path),
                                 "reference_files": reference_paths, "source_files_used": [str(p) for p in observed.used_paths],
                                 "fully_synthetic": False, "placement": canvas.placement.to_dict()})
                            done += 1
                            self._record_output(record, result, manifest)
                    except (GenerationCancelled, PipelineCancelled, BackendFatalError):
                        raise
                    except (OSError, ValueError, RuntimeError) as exc:
                        self._record_error(exc, result, manifest, str(analysis.path))
                        if not self.config.continue_on_error:
                            raise
                for index in range(variants):
                    self._check_cancelled()
                    references, reference_paths = self._references(ordered[0], ordered)
                    canvas = prepare_variant_canvas(frame=self.config.variant_frame, target_long_edge=self.config.target_long_edge)
                    prompt, negative = self._prompts(subject, True, index)
                    record = self._generate(backend, canvas, references, prompt, negative,
                        f"subject_{group_index:03d}__synthetic_{index + 1:02d}",
                        self._seed_for(group_index, 10000, index), done, total, "Synthetic view",
                        {"kind": "synthetic_variant", "reference_files": reference_paths, "fully_synthetic": True})
                    done += 1
                    self._record_output(record, result, manifest)
            if not result.output_paths:
                raise BackendFatalError("No reconstruction was produced. Diagnostics are not result images. See bodybuilder.log.")
            manifest["status"] = "completed_with_errors" if result.errors else "completed"
            self.callbacks.progress(total, total, f"Saved {len(result.output_paths)} image(s)")
        except (PipelineCancelled, GenerationCancelled):
            result.cancelled = True
            manifest["status"] = "cancelled"
            self._log("Cancelled. Completed images have been kept.")
        except Exception as exc:
            # Application boundary: record unexpected programming faults and re-raise.
            manifest["status"] = "failed"
            self._record_error(exc, result, manifest, "run")
            raise
        finally:
            for backend in self._backends.values():
                try:
                    backend.close()
                except (RuntimeError, OSError) as exc:
                    self._log(f"Backend cleanup failed: {exc}")
            if self._upscaler is not None:
                try:
                    self._upscaler.close()
                except (RuntimeError, OSError) as exc:
                    self._log(f"Upscaler cleanup failed: {exc}")
            manifest["finished_at"] = datetime.now().astimezone().isoformat()
            try:
                write_json(result.manifest_path, manifest)
            finally:
                self.logger.removeHandler(handler)
                handler.close()
        return result

    def _groups(self, analyses: list[ImageAnalysis]) -> list[list[ImageAnalysis]]:
        if self.config.group_all_images:
            return [analyses]
        from bodybuilder.core.clustering import group_images
        grouped = group_images(analyses, group_all=False, threshold=self.config.grouping_similarity_threshold,
            model_id=self.config.model_settings.grouping_model_id, device=self.config.device, log=self._log)
        if grouped.warning:
            self._log(grouped.warning)
        return grouped.groups

    def _references(self, anchor: ImageAnalysis, ordered: list[ImageAnalysis]) -> tuple[tuple[Image.Image, ...], list[str]]:
        chosen = [anchor, *(item for item in ordered if item.path != anchor.path)][:16]
        if len(ordered) > len(chosen):
            self._log(f"Using {len(chosen)} references for this photo out of {len(ordered)}; the source is always included.")
        images = []
        for item in chosen:
            self._check_cancelled()
            image, mask = load_fragment(item.path)
            images.append(image.crop(mask.getbbox()))
        return tuple(images), [str(item.path) for item in chosen]

    def _prompts(self, subject: SubjectKind, synthetic: bool, index: int) -> tuple[str, str]:
        from bodybuilder.core.prompts import completion_prompt, variant_prompt
        if synthetic:
            _slug, prompt, negative = variant_prompt(subject, index, self.config.variant_frame, self.config.custom_prompt)
        else:
            prompt, negative = completion_prompt(subject, self.config.custom_prompt)
        return prompt, negative + ", collage, contact sheet, grid, panels, blank blocks, black square, white square"

    def _generate(self, backend: GenerationBackend, canvas: PreparedCanvas,
                  references: tuple[Image.Image, ...], prompt: str, negative: str, slug: str,
                  seed: int, done: int, total: int, label: str, metadata: dict[str, Any]) -> dict[str, Any]:
        self._check_cancelled()
        request = GenerationRequest(canvas.image, canvas.generated_mask, references[0], None,
            prompt, negative, seed, self.config.inference_steps, self.config.guidance_scale,
            1.0 if metadata["fully_synthetic"] else self.config.denoising_strength,
            self.config.reference_fidelity, canvas.image.width, canvas.image.height,
            fully_synthetic=metadata["fully_synthetic"], reference_images=references)
        self.callbacks.progress(done, total, label)
        generated = backend.generate(request, cancel_event=self.cancel_event,
            progress=lambda step, steps, message: self.callbacks.progress(done, total, f"{label}: {message} {step}/{steps}"))
        self._check_cancelled()
        if self.config.backend == BackendKind.SDXL:
            validate_generation(generated, canvas.image, canvas.generated_mask)
        reconstructed = preserve_observed_pixels(generated, canvas.image, canvas.generated_mask)
        assert self.run_dir is not None
        output_folder = "images" if self.config.backend == BackendKind.SDXL else "diagnostic_previews"
        record = self._save_output(image=reconstructed, source_canvas=canvas.image,
            observed_mask=canvas.observed_mask, generated_mask=canvas.generated_mask,
            output_base=self.run_dir / output_folder / slug,
            metadata={**metadata, "seed": seed, "prompt": prompt, "negative_prompt": negative,
                      "backend": backend.name, "model": backend.model_identifier,
                      "generated_fraction": generated_fraction(canvas.generated_mask),
                      "technical_validation": "passed" if self.config.backend == BackendKind.SDXL else "diagnostic_only",
                      "generation": getattr(backend, "last_generation_metadata", {})})
        return record

    def _record_output(self, record: dict[str, Any], result: PipelineRunResult, manifest: dict[str, Any]) -> None:
        result.output_paths.append(Path(record["image"]))
        manifest["outputs"].append(record)
        write_json(result.manifest_path, manifest)
        self.callbacks.preview(Path(record["image"]))

    def _prepare_observed_source(self, analysis: ImageAnalysis, group: list[ImageAnalysis]) -> StitchResult:
        image, mask = load_fragment(analysis.path)
        if self.config.stitch_overlaps and self.config.max_stitch_candidates and mask.getextrema() == (255, 255):
            from bodybuilder.core.stitching import stitch_from_anchor
            candidates = [item.path for item in group if item.path != analysis.path
                          and load_fragment(item.path)[1].getextrema() == (255, 255)]
            return stitch_from_anchor(analysis.path, candidates,
                max_candidates=self.config.max_stitch_candidates, log=self._log)
        return StitchResult(image, mask, [analysis.path])

    def _save_output(self, *, image: Image.Image, source_canvas: Image.Image,
                     observed_mask: Image.Image, generated_mask: Image.Image,
                     output_base: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        final, source = image.convert("RGB"), source_canvas.convert("RGB")
        observed, missing = observed_mask.convert("L"), generated_mask.convert("L")
        enhancement = "none"
        if self.config.upscale_2x:
            from bodybuilder.ai.upscale import LanczosUpscaler, Swin2SRUpscaler
            if self._upscaler is None:
                self._upscaler = Swin2SRUpscaler(model_id=self.config.model_settings.upscaler_model_id,
                    device=self.config.device, log=self._log)
            try:
                final = self._upscaler.upscale(final)
                enhancement = self._upscaler.name
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self._log(f"AI enhancement unavailable: {exc}. Using deterministic Lanczos resizing.")
                self._upscaler.close()
                self._upscaler = LanczosUpscaler()
                final = self._upscaler.upscale(image)
                enhancement = self._upscaler.name
            source = source.resize(final.size, Image.Resampling.LANCZOS)
            observed = observed.resize(final.size, Image.Resampling.NEAREST)
            missing = missing.resize(final.size, Image.Resampling.NEAREST)
        final.paste(source, (0, 0), observed)
        diagnostics = (self.run_dir or output_base.parent) / "diagnostics" / output_base.name
        # Dots in source names must not erase the completion suffix.
        image_path = output_base.parent / (output_base.name + ".png")
        record = {**metadata, "image": str(image_path), "enhancement": enhancement,
            "observed_mask": str(diagnostics / "observed_mask.png"),
            "generated_mask": str(diagnostics / "generated_mask.png"),
            "observed_working_pixels_preserved_exactly": observed.getbbox() is not None and not self.config.upscale_2x,
            "observed_region_processing": "2x Lanczos resampling; no AI changes" if self.config.upscale_2x else "Copied exactly from working canvas",
            "output_width": final.width, "output_height": final.height}
        save_png(source, diagnostics / "source_canvas.png")
        save_png(observed, diagnostics / "observed_mask.png")
        save_png(missing, diagnostics / "generated_mask.png")
        write_json(diagnostics / "metadata.json", record)
        save_png(final, image_path, record)
        self._log(f"Saved result: {image_path}")
        return record

    def _backend_for(self, subject_kind: SubjectKind, *, use_face_adapter: bool) -> GenerationBackend:
        key = (subject_kind, use_face_adapter)
        if key not in self._backends:
            for backend in self._backends.values():
                backend.close()
            self._backends.clear()
            self._backends[key] = ClassicalFillBackend() if self.config.backend == BackendKind.CLASSICAL else SdxlBackend(
                subject_kind=subject_kind, device=self.config.device, models=self.config.model_settings,
                use_face_adapter=use_face_adapter, log=self._log)
        return self._backends[key]

    def _resolve_subject_kind(self, group: list[ImageAnalysis]) -> SubjectKind:
        # No face detection is not evidence of an object: cropped faces often evade detection.
        return SubjectKind.PERSON if self.config.subject_kind == SubjectKind.AUTO else self.config.subject_kind

    def _seed_for(self, subject_index: int, source_index: int, variant_index: int) -> int:
        return (self.config.seed + subject_index * 1000003 + source_index * 10007 + variant_index * 101) % (2**32 - 1)

    def _record_error(self, exc: Exception, result: PipelineRunResult, manifest: dict[str, Any], source: str) -> None:
        error = {"source": source, "type": type(exc).__name__, "message": str(exc)}
        result.errors.append(error)
        manifest["errors"].append(error)
        if self.logger:
            self.logger.exception("Reconstruction failed: %s", error)
        self.callbacks.log(f"ERROR: {source}: {exc}")

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelled("Cancelled")

    def _validate_paths(self) -> None:
        source, output = self.config.input_dir.resolve(), self.config.output_dir.resolve()
        if not source.is_dir():
            raise ValueError(f"Source folder does not exist: {source}")
        if output == source or source in output.parents:
            raise ValueError("Choose an output folder outside the source folder")
        output.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        self.callbacks.log(message)
