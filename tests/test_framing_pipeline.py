"""Framing failure regressions with controlled vision and generation backends.

Real image I/O and the whole pipeline run. These do not test visual fidelity.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bodybuilder.ai.base import BackendFatalError
from bodybuilder.ai.subject_vision import Region
from bodybuilder.config import PipelineConfig
from bodybuilder.core.joint_reconstruction import (
    EvidenceConsistencyError,
    JointReconstructionPipeline,
    JointRunResult,
)
from bodybuilder.core.subject_evidence import SubjectEvidence


def texture(size, seed=42):
    values = np.random.default_rng(seed).integers(15, 240, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(values)


@pytest.fixture
def controlled_pipeline(tmp_path, monkeypatch):
    state = SimpleNamespace(calls=[], checks=[], fallback="edge", fail_at=None,
                            cancel_at_check=False, eyewear=False)

    class Vision:
        def __init__(self, event, log):
            self.event = event

        def detect(self, image):
            return [Region("bicycle", (0, 0, *image.size))]

        def locate(self, image, label):
            if state.cancel_at_check:
                self.event.set()
            mode = state.checks.pop(0) if state.checks else state.fallback
            if mode == "none":
                return []
            if mode == "multiple":
                return [Region(label, (10, 20, 240, 490)), Region(label, (270, 20, 500, 490))]
            if mode == "passed":
                return [Region(label, (50, 50, image.width - 50, image.height - 50))]
            return [Region(label, (0, 0, image.width, image.height))]

        def describe_regions(self, image):
            return [Region("sunglasses", (10, 10, 100, 100))] if state.eyewear else []

        def close(self):
            pass

    class Backend:
        def __init__(self, **kwargs):
            self.last_generation_metadata = {}

        def prepare(self):
            pass

        def generate(self, request, **kwargs):
            state.calls.append(request)
            if state.fail_at == len(state.calls):
                raise BackendFatalError("GPU memory exhausted")
            self.last_generation_metadata = {"effective_seed": request.seed}
            return texture(request.canvas.size, request.seed)

        def close(self):
            pass

    root = "bodybuilder.core.joint_reconstruction"
    monkeypatch.setattr(root + ".SubjectVision", Vision)
    monkeypatch.setattr(root + ".EvidenceSdxlBackend", Backend)
    monkeypatch.setattr(root + ".build_evidence",
        lambda views, vision, **kwargs: SubjectEvidence("bicycle", views))
    source = tmp_path / "input"
    source.mkdir()
    texture((128, 160)).save(source / "original.png")
    pipeline = JointReconstructionPipeline(PipelineConfig(source, tmp_path / "out", target_long_edge=512))
    return pipeline, state


def manifest(result):
    return json.loads(result.manifest_path.read_text())


def test_repeated_edge_warning_keeps_output_and_continues_other_views(controlled_pipeline):
    pipeline, state = controlled_pipeline
    pipeline.config.synthetic_variants = 1
    result = pipeline.run()
    report = manifest(result)
    assert len(state.calls) == 4  # two bounded layout passes for each of two views
    assert len(result.output_paths) == len(result.review_paths) == 2
    assert not result.errors
    assert report["status"] == "completed_with_warnings"
    assert report["review_required"] and report["review_count"] == 2
    for record, path in zip(report["outputs"], result.output_paths, strict=True):
        assert path.is_file() and path.stem.endswith("__needs_review")
        assert record["framing_check"] == "needs_review"
        assert record["review_required"] and record["warnings"]
        assert record["fully_synthetic"] and record["complete_geometry"] == "not_verified"
        with Image.open(record["observed_mask"]) as observed:
            assert observed.getextrema() == (0, 0)
        with Image.open(path) as image:
            assert image.size == (512, 512)


def test_edge_recovery_uses_real_masked_generation_not_a_border_only_pass(controlled_pipeline):
    pipeline, state = controlled_pipeline
    state.checks = ["edge", "passed", "passed"]
    result = pipeline.run()
    assert len(state.calls) == 2
    initial, recovery = state.calls
    assert initial.free_composition and not recovery.free_composition
    assert recovery.reference_images == initial.reference_images
    assert all(image.size == (128, 160) for image in recovery.reference_images)
    assert recovery.generated_mask.getextrema() == (0, 255)
    assert recovery.canvas.size == initial.canvas.size == (512, 512)
    record = manifest(result)["outputs"][0]
    assert record["framing_check"] == "bounding_box_margin_passed"
    assert record["selected_draft"] == 2
    assert not record["review_required"] and not result.review_paths
    assert record["framing_history"][1]["stage"] == "outpaint_generated_draft"
    assert record["framing_history"][1]["draft_is_generated_not_observed"]
    with Image.open(result.output_paths[0]) as image:
        protected = np.asarray(recovery.generated_mask) == 0
        assert np.array_equal(np.asarray(image)[protected], np.asarray(recovery.canvas)[protected])


@pytest.mark.parametrize("mode", ["none", "multiple"])
def test_uncertain_detector_results_are_not_reported_as_verified(controlled_pipeline, mode):
    pipeline, state = controlled_pipeline
    state.fallback = mode
    result = pipeline.run()
    assert len(state.calls) == 2 and all(call.free_composition for call in state.calls)
    assert result.review_paths
    record = manifest(result)["outputs"][0]
    assert record["framing_check"] == "needs_review"
    assert record["framing_history"][-1]["status"] in {"not_localized", "multiple_subjects"}


def test_post_refinement_edge_check_is_also_nonfatal(controlled_pipeline, monkeypatch):
    pipeline, state = controlled_pipeline
    state.checks = ["passed", "edge"]
    refined = []

    def refine(image, *args):
        refined.append(True)
        return image, []

    monkeypatch.setattr(pipeline, "_refine_parts", refine)
    result = pipeline.run()
    assert refined and len(state.calls) == 1
    assert manifest(result)["status"] == "completed_with_warnings"
    assert manifest(result)["outputs"][0]["framing_history"][-1]["stage"] == "after_regional_refinement"


def test_worse_recovery_does_not_replace_better_draft(controlled_pipeline):
    pipeline, state = controlled_pipeline
    state.checks = ["edge", "none", "edge"]
    result = pipeline.run()
    record = manifest(result)["outputs"][0]
    assert record["selected_draft"] == 1
    with Image.open(result.output_paths[0]) as image:
        assert np.array_equal(np.asarray(image), np.asarray(texture(image.size, state.calls[0].seed)))


def test_cancellation_at_framing_check_does_not_start_recovery(controlled_pipeline):
    pipeline, state = controlled_pipeline
    state.cancel_at_check = True
    result = pipeline.run()
    assert result.cancelled and not result.output_paths
    assert len(state.calls) == 1
    assert manifest(result)["status"] == "cancelled"
    assert list((result.run_dir / "diagnostics").rglob("draft_01.png"))


def test_real_backend_failure_is_not_swallowed_and_prior_output_survives(controlled_pipeline):
    pipeline, state = controlled_pipeline
    pipeline.config.synthetic_variants = 1
    state.checks = ["passed", "passed", "edge"]
    state.fail_at = 3
    with pytest.raises(BackendFatalError, match="GPU memory exhausted"):
        pipeline.run()
    report = json.loads((pipeline.run_dir / "run_manifest.json").read_text())
    assert report["status"] == "failed"
    assert len(report["outputs"]) == 1
    assert len(list((pipeline.run_dir / "images").glob("*.png"))) == 1
    assert len(state.calls) == 3


def test_eyewear_consistency_error_remains_distinct(controlled_pipeline, monkeypatch):
    pipeline, state = controlled_pipeline
    state.fallback, state.eyewear = "passed", True
    monkeypatch.setattr("bodybuilder.core.joint_reconstruction.full_subject_prompt",
                        lambda *args: ("bicycle", "sunglasses"))
    with pytest.raises(EvidenceConsistencyError, match="Eyewear"):
        pipeline.run()
    assert len(state.calls) == 2  # existing semantic retry is still bounded


def test_source_repair_does_not_reframe_observed_pixels(controlled_pipeline):
    pipeline, state = controlled_pipeline
    pipeline.complete_subject = False
    result = pipeline.run()
    assert len(state.calls) == 1 and not state.calls[0].free_composition
    record = manifest(result)["outputs"][0]
    assert record["framing_check"] == "source_view_preserved"
    assert not result.review_paths
    with Image.open(record["observed_mask"]) as observed, Image.open(result.output_paths[0]) as final:
        chosen = np.asarray(observed) > 127
        assert chosen.any()
        assert np.array_equal(np.asarray(final)[chosen], np.asarray(state.calls[0].canvas)[chosen])


@pytest.mark.parametrize("cancelled", [False, True])
def test_ui_reports_review_instead_of_success_or_error(tmp_path, monkeypatch, cancelled):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from bodybuilder.ui.subject_window import MainWindow

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "images" / "subject__whole_01__needs_review.png"
    path.parent.mkdir()
    texture((64, 64)).save(path)
    result = JointRunResult(tmp_path, tmp_path / "run_manifest.json", output_paths=[path],
        cancelled=cancelled, review_paths=[str(path)], review_messages=["Subject may be cropped"])
    window = MainWindow()
    try:
        window._add_result(str(path))
        window._completed(result)
        app.processEvents()
        assert "need framing review" in window.status.text()
        assert ("Cancelled" if cancelled else "Completed with warnings") in window.status.text()
        assert "framing needs review" in window.results.item(0).text()
        assert window.details_toggle.isChecked()
        assert window._error_dialog is None
        assert window.open_output_button.isEnabled()
    finally:
        window.close()
