"""Exercise real decoding, analysis and prompts; only the generative model is fake."""

from __future__ import annotations

import json
import threading

import numpy as np
from PIL import Image

from bodybuilder.ai.base import GenerationBackend
from bodybuilder.config import PipelineConfig
from bodybuilder.core.pipeline import ReconstructionPipeline


class ImageBackend(GenerationBackend):
    def generate(self, request, **kwargs):
        pixels = np.random.default_rng(42).integers(
            20, 230, (request.height, request.width, 3), dtype=np.uint8
        )
        return Image.fromarray(pixels)


def test_real_source_analysis_skips_corrupt_files_and_completes_fragments(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    image = Image.new("RGBA", (80, 100), (40, 100, 180, 255))
    image.paste((0, 0, 0, 0), (40, 0, 80, 100))
    image.save(source / "fragment.png")
    (source / "corrupt.jpg").write_bytes(b"not an image")
    pipeline = ReconstructionPipeline(PipelineConfig(source, tmp_path / "output"))
    monkeypatch.setattr(pipeline, "_backend_for", lambda *args, **kwargs: ImageBackend())
    result = pipeline.run()
    manifest = json.loads(result.manifest_path.read_text())
    assert len(result.output_paths) == 1
    assert len(result.errors) == 1
    assert manifest["status"] == "completed_with_errors"
    record = manifest["outputs"][0]
    assert record["kind"] == "source_completion"
    assert "prompt" in record
    with Image.open(record["observed_mask"]) as mask:
        assert mask.getextrema() == (0, 255)
    with Image.open(record["image"]) as output:
        assert output.mode == "RGB"
    assert len(list((result.run_dir / "images").iterdir())) == 1


def test_cancellation_after_loading_keeps_cancelled_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (80, 100), (20, 100, 180)).save(source / "fragment.png")
    cancelled = threading.Event()

    class CancellingBackend(ImageBackend):
        def prepare(self):
            cancelled.set()

    pipeline = ReconstructionPipeline(
        PipelineConfig(source, tmp_path / "output"), cancel_event=cancelled
    )
    monkeypatch.setattr(pipeline, "_backend_for", lambda *args, **kwargs: CancellingBackend())
    result = pipeline.run()
    assert result.cancelled
    assert not result.output_paths
    assert json.loads(result.manifest_path.read_text())["status"] == "cancelled"
