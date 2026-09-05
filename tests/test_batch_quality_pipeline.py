"""Real orchestration/export and image masks with controlled model calls.

These are regression checks, not a visual-quality benchmark of pretrained SDXL.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bodybuilder.ai.base import BackendFatalError
from bodybuilder.ai.subject_vision import Region
from bodybuilder.config import PipelineConfig
from bodybuilder.core.joint_reconstruction import JointReconstructionPipeline
from bodybuilder.core.run_options import QualityMode, WorkflowMode, quality_profile
from bodybuilder.core.subject_evidence import PartEvidence, SourceView, SubjectEvidence


def texture(size=(80, 100), seed=1):
    return Image.fromarray(np.random.default_rng(seed).integers(15, 235, (size[1], size[0], 3), dtype=np.uint8))


@pytest.fixture
def controlled(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], evidence_calls=0, loads=0, fail_at=None)

    class Vision:
        def __init__(self, *args):
            pass
        def detect(self, image):
            return [Region("bicycle", (10, 10, image.width - 10, image.height - 10))]
        def locate(self, image, label):
            return [Region(label, (20, 20, image.width - 20, image.height - 20))]
        def describe_regions(self, image):
            return []
        def check_cancelled(self):
            pass
        def close(self):
            pass

    class Backend:
        def __init__(self, **kwargs):
            self.last_generation_metadata = {}
        def prepare(self):
            state.loads += 1
        def generate(self, request, **kwargs):
            state.calls.append(request)
            if state.fail_at == len(state.calls):
                raise BackendFatalError("Test model failure")
            return texture(request.canvas.size, seed=len(state.calls))
        def close(self):
            pass

    def build(views, *args, **kwargs):
        state.evidence_calls += 1
        return SubjectEvidence("bicycle", views)

    root = "bodybuilder.core.joint_reconstruction"
    monkeypatch.setattr(root + ".SubjectVision", Vision)
    monkeypatch.setattr(root + ".EvidenceSdxlBackend", Backend)
    monkeypatch.setattr(root + ".build_evidence", build)
    monkeypatch.setattr(root + ".assemble_overlap", lambda *args: None)
    folder = tmp_path / "originals"
    folder.mkdir()
    for i in range(2):
        texture(seed=i).save(folder / f"source_{i}.png")
    return folder, state


class TestUpscaler:
    """Deliberately destroys every pixel; export must restore real evidence."""
    __test__ = False
    name = "test-destructive-2x"

    def upscale(self, image):
        return Image.new("RGB", (image.width * 2, image.height * 2), "magenta")
    def close(self):
        pass


def test_combined_run_produces_each_source_and_exactly_five_views(controlled, tmp_path):
    source, state = controlled
    original_bytes = {path: path.read_bytes() for path in source.glob("*.png")}
    config = PipelineConfig(source, tmp_path / "output", target_long_edge=512,
                            synthetic_variants=5, upscale_2x=True)
    pipeline = JointReconstructionPipeline(config, workflow=WorkflowMode.BOTH)
    pipeline._upscaler = TestUpscaler()
    result = pipeline.run()
    manifest = json.loads(result.manifest_path.read_text())
    assert state.evidence_calls == state.loads == 1
    assert len(result.restored_paths) == 2 and len(result.synthetic_paths) == 5
    assert len(result.output_paths) == 7
    assert manifest["planned_outputs"] == manifest["output_counts"] == {"restored_sources": 2, "synthetic_views": 5}
    assert manifest["workflow"] == WorkflowMode.BOTH.value
    assert manifest["status"] == "completed"
    assert [call.free_composition for call in state.calls] == [False, False] + [True] * 5
    assert len({call.seed for call in state.calls}) == 7
    for path, before in original_bytes.items():
        assert path.read_bytes() == before
    for record in manifest["outputs"]:
        assert record["output_width"] == record["output_height"] or not record["fully_synthetic"]
        with Image.open(record["image"]) as final, Image.open(record["observed_mask"]) as observed:
            assert max(final.size) == 1024
            mask = np.asarray(observed) > 127
            if not record["fully_synthetic"]:
                assert mask.any() and Path(record["source_file"]) in original_bytes
                with Image.open(Path(record["observed_mask"]).parent / "source_canvas.png") as working:
                    assert np.array_equal(np.asarray(final)[mask], np.asarray(working)[mask])
            else:
                assert not mask.any() and record["source_file"] is None
        # The generated restored images must never become synthetic references.
        assert all(Path(ref["source"]) in original_bytes for ref in record["reference_evidence"])


@pytest.mark.parametrize("workflow,sources,synthetic", [
    (WorkflowMode.SOURCES, 2, 0), (WorkflowMode.SYNTHETIC, 0, 5),
])
def test_independent_workflows_have_exact_counts(controlled, tmp_path, workflow, sources, synthetic):
    source, _state = controlled
    config = PipelineConfig(source, tmp_path / "out", target_long_edge=512, synthetic_variants=5)
    result = JointReconstructionPipeline(config, workflow=workflow).run()
    assert len(result.restored_paths) == sources and len(result.synthetic_paths) == synthetic


@pytest.mark.parametrize("mode", list(QualityMode))
def test_profile_reaches_generation_and_detail_requests(controlled, tmp_path, mode):
    source, state = controlled
    config = PipelineConfig(source, tmp_path / "out", synthetic_variants=1)
    pipeline = JointReconstructionPipeline(config, workflow=WorkflowMode.SYNTHETIC, quality=mode)
    result = pipeline.run()
    profile = quality_profile(mode)
    main = state.calls[0]
    assert main.steps == profile.generation_steps
    assert max(main.canvas.size) == profile.working_long_edge
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["quality_profile"] == profile.report()
    assert manifest["outputs"][0]["quality_profile"] == profile.report()
    # Give every mode the same original details. Only the processing budget varies.
    image = texture((128, 128))
    view = SourceView(source / "source_0.png", image, Image.new("L", image.size, 255), [])
    parts = [PartEvidence(f"detail_{i}", view.path, (20, 20, 80, 80), image.crop((20, 20, 80, 80)), 1)
             for i in range(10)]
    pipeline.evidence = SubjectEvidence("bicycle", [view], parts=parts)
    allowed = Image.new("L", image.size, 255)
    allowed.paste(0, (0, 0, 30, 128))
    state.calls.clear()
    final, records = pipeline._refine_parts(image, allowed, "bicycle", "collage", 1, 0, 1)
    assert len(state.calls) == len(records) == profile.max_detail_passes
    assert all(call.steps == profile.detail_steps for call in state.calls)
    assert all(call.reference_images[0] in [part.image for part in parts] for call in state.calls)
    known = np.asarray(allowed) == 0
    assert np.array_equal(np.asarray(final)[known], np.asarray(image)[known])


def test_failure_in_synthetic_stage_keeps_completed_source_restorations(controlled, tmp_path):
    source, state = controlled
    state.fail_at = 3
    config = PipelineConfig(source, tmp_path / "out", target_long_edge=512, synthetic_variants=5)
    pipeline = JointReconstructionPipeline(config, workflow=WorkflowMode.BOTH)
    with pytest.raises(BackendFatalError, match="Test model failure"):
        pipeline.run()
    report = json.loads((pipeline.run_dir / "run_manifest.json").read_text())
    assert report["status"] == "failed"
    assert report["output_counts"] == {"restored_sources": 2, "synthetic_views": 0}
    assert len(report["outputs"]) == 2
    assert all(Path(record["image"]).is_file() for record in report["outputs"])


def test_legacy_programmatic_counts_and_custom_resolution_remain_compatible(tmp_path):
    config = PipelineConfig(tmp_path, tmp_path / "out", target_long_edge=512, synthetic_variants=2)
    pipeline = JointReconstructionPipeline(config)
    assert pipeline.config.target_long_edge == 512
    assert pipeline._build_tasks(["a"]) == [None, None, None]
    pipeline.complete_subject = False
    assert pipeline._build_tasks(["a"]) == ["a", None, None]
