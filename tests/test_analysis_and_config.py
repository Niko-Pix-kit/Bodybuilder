from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bodybuilder.config import PipelineConfig
from bodybuilder.core.analysis import best_face_crops
from bodybuilder.core.clustering import _connected_components
from bodybuilder.core.types import FaceBox, ImageAnalysis


def _analysis(path: Path, quality: float) -> ImageAnalysis:
    return ImageAnalysis(
        path=path,
        width=200,
        height=200,
        file_size=100,
        blur_score=quality,
        exposure_score=quality,
        detail_score=quality,
        resolution_score=quality,
        quality_score=quality,
        faces=(FaceBox(50, 45, 80, 90),),
    )


def test_best_face_crops_iterates_over_all_analyses(tmp_path: Path) -> None:
    low = tmp_path / "low.png"
    high = tmp_path / "high.png"
    Image.new("RGB", (200, 200), "red").save(low)
    Image.new("RGB", (200, 200), "blue").save(high)

    crops = best_face_crops([_analysis(low, 0.2), _analysis(high, 0.9)], limit=2)
    assert len(crops) == 2
    assert crops[0].getpixel((0, 0)) == (0, 0, 255)


def test_connected_components_use_transitive_similarity() -> None:
    similarity = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.8, 0.1],
            [0.2, 0.8, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]
    )
    assert _connected_components(similarity, 0.7) == [[0, 1, 2], [3]]


def test_configuration_serializes_paths(tmp_path: Path) -> None:
    config = PipelineConfig(input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    payload = config.to_dict()
    assert payload["input_dir"] == str(tmp_path / "input")
    assert payload["model_settings"]["inpainting_model_id"]


def test_configuration_rejects_invalid_target_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        PipelineConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            target_long_edge=1025,
        )
