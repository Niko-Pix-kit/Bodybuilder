from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from bodybuilder.ai.upscale import Upscaler
from bodybuilder.config import BackendKind, PipelineConfig
from bodybuilder.core.pipeline import ReconstructionPipeline
from bodybuilder.core.stitching import _estimate_candidate_to_anchor, _merge


class DestructiveUpscaler(Upscaler):
    name = "destructive-test-upscaler"

    def upscale(self, image: Image.Image) -> Image.Image:
        return Image.new("RGB", (image.width * 2, image.height * 2), (255, 0, 255))


def test_save_output_restores_observed_pixels_after_upscale(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        backend=BackendKind.CLASSICAL,
        upscale_2x=True,
    )
    pipeline = ReconstructionPipeline(config)
    pipeline._upscaler = DestructiveUpscaler()

    source = Image.new("RGB", (32, 32), (10, 20, 30))
    result = Image.new("RGB", (32, 32), (100, 110, 120))
    observed = Image.new("L", (32, 32), 0)
    observed.paste(255, (8, 8, 24, 24))
    generated = Image.eval(observed, lambda value: 255 - value)
    record = pipeline._save_output(
        image=result,
        source_canvas=source,
        observed_mask=observed,
        generated_mask=generated,
        output_base=output_dir / "sample",
        metadata={"kind": "test"},
    )

    with Image.open(record["image"]) as saved_image:
        saved = np.asarray(saved_image.convert("RGB"))
    scaled_observed = np.asarray(observed.resize((64, 64), Image.Resampling.NEAREST)) > 127
    scaled_source = np.asarray(source.resize((64, 64), Image.Resampling.LANCZOS))
    assert np.array_equal(saved[scaled_observed], scaled_source[scaled_observed])
    assert record["observed_region_processing"].startswith("2x Lanczos")


def _textured_scene() -> Image.Image:
    rng = np.random.default_rng(123)
    array = rng.integers(0, 256, size=(300, 500, 3), dtype=np.uint8)
    for index in range(12):
        cv2.circle(array, (30 + index * 35, 40 + (index % 4) * 60), 12, (255, 255, 255), 2)
        cv2.putText(
            array,
            str(index),
            (20 + index * 35, 250 - (index % 3) * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
    return Image.fromarray(array, mode="RGB")


def test_overlap_estimation_and_non_destructive_merge() -> None:
    scene = _textured_scene()
    anchor = scene.crop((0, 0, 340, 300))
    candidate = scene.crop((160, 0, 500, 300))
    estimate = _estimate_candidate_to_anchor(anchor, candidate)
    assert estimate is not None
    transform, _, _ = estimate
    merged = _merge(anchor, Image.new("L", anchor.size, 255), candidate, transform)
    assert merged is not None
    image, observed = merged
    assert image.width > anchor.width
    assert np.asarray(observed, dtype=np.float32).mean() > 240
