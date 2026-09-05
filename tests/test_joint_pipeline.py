"""Integration with controlled detectors/backends; not a visual fidelity test."""
import json
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from bodybuilder.ai.subject_vision import Region
from bodybuilder.config import PipelineConfig, SubjectKind
from bodybuilder.core.joint_reconstruction import JointReconstructionPipeline
from bodybuilder.core.subject_evidence import PartEvidence, SourceView, SubjectEvidence


def texture(size):
    return Image.fromarray(np.random.default_rng(31).integers(15, 240, (size[1], size[0], 3), dtype=np.uint8))


class Vision:
    def __init__(self, *args):
        pass
    def detect(self, image):
        return [Region("bicycle", (20, 20, image.width - 20, image.height - 20))]
    def describe_regions(self, image):
        return []
    def locate(self, image, label):
        return [Region(label, (20, 20, image.width - 20, image.height - 20))]
    def check_cancelled(self):
        pass
    def close(self):
        pass


class Backend:
    last_generation_metadata = {}
    calls = []
    def __init__(self, **kwargs):
        pass
    def prepare(self):
        pass
    def generate(self, request, **kwargs):
        self.calls.append(request)
        return texture(request.canvas.size)
    def close(self):
        pass


def test_whole_subject_not_one_outpaint_per_fragment(tmp_path, monkeypatch):
    root = "bodybuilder.core.joint_reconstruction"
    monkeypatch.setattr(root + ".SubjectVision", Vision)
    monkeypatch.setattr(root + ".EvidenceSdxlBackend", Backend)
    source = tmp_path / "input"
    source.mkdir()
    for name in ("part_a.png", "part_b.png"):
        texture((160, 220)).save(source / name)
    config = PipelineConfig(source, tmp_path / "output", subject_kind=SubjectKind.AUTO,
                            target_long_edge=512, synthetic_variants=1)
    Backend.calls = []
    pipeline = JointReconstructionPipeline(config)
    result = pipeline.run()
    assert len(result.output_paths) == 2
    assert all(request.free_composition for request in Backend.calls)
    assert all(request.generated_mask.getextrema() == (255, 255) for request in Backend.calls)
    assert all(request.canvas.getpixel((0, 0)) == (127, 127, 127) for request in Backend.calls)
    assert all(request.width == request.height for request in Backend.calls)
    report = json.loads(result.manifest_path.read_text())
    assert report["workflow"] == "whole_subject"
    assert all(record["fully_synthetic"] for record in report["outputs"])
    assert len(report["sources"]) == 2
    assert (result.run_dir / "subject_evidence.json").is_file()
    assert all(path.parent.name == "images" for path in result.output_paths)


def test_regional_pass_is_scoped_and_never_touches_observed_pixels(tmp_path):
    pipeline = JointReconstructionPipeline(PipelineConfig(tmp_path, tmp_path / "output"))
    pipeline.vision = Vision()
    pipeline.backend = Backend()
    original = texture((128, 128))
    view = SourceView(Path("original.png"), original, Image.new("L", original.size, 255), [])
    part = PartEvidence("wheel", view.path, (0, 0, 30, 30), original.crop((0, 0, 30, 30)), 1)
    pipeline.evidence = SubjectEvidence("bicycle", [view], parts=[part])
    allowed = Image.new("L", original.size, 0)
    allowed.paste(255, (50, 50, 100, 100))
    Backend.calls = []
    final, records = pipeline._refine_parts(original, allowed, "bicycle", "collage", 42, 0, 1)
    assert len(Backend.calls) == 1
    assert len(Backend.calls[0].reference_masks) == 1
    assert Backend.calls[0].reference_images == (part.image,)
    known = np.asarray(allowed) == 0
    assert np.array_equal(np.asarray(final)[known], np.asarray(original)[known])
    assert records[0]["source"] == "original.png"
    assert records[0]["pixel_copy"] is False


def test_new_request_fields_do_not_change_existing_configuration(tmp_path):
    from bodybuilder.ai.evidence_sdxl import EvidenceRequest
    image = texture((64, 64))
    request = EvidenceRequest(image, Image.new("L", image.size, 255), image, None,
        "whole object", "cropped", 1, 20, 5, 1, 0.6, 64, 64, reference_images=(image,), free_composition=True)
    assert request.free_composition
    assert not request.reference_masks
    assert "input_dir" in asdict(PipelineConfig(tmp_path, tmp_path / "out"))


def test_cancellation_before_model_download(tmp_path):
    import pytest
    from bodybuilder.ai.base import GenerationCancelled
    from bodybuilder.ai.subject_vision import SubjectVision
    event = threading.Event()
    event.set()
    with pytest.raises(GenerationCancelled):
        SubjectVision(event).prepare()
