from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from bodybuilder.config import CompletionAspect, VariantFrame
from bodybuilder.core.canvas import (
    prepare_outpaint_canvas,
    prepare_variant_canvas,
    preserve_observed_pixels,
)
from bodybuilder.core.image_io import load_rgb, save_png, scan_images, write_json


def test_preserve_observed_pixels_restores_source_exactly() -> None:
    source = Image.new("RGB", (64, 48), (10, 20, 30))
    mask = Image.new("L", source.size, 255)
    canvas = prepare_outpaint_canvas(
        source,
        mask,
        aspect=CompletionAspect.LANDSCAPE,
        margin_percent=50,
        target_long_edge=512,
    )
    generated = Image.new("RGB", canvas.image.size, (200, 100, 50))
    result = preserve_observed_pixels(generated, canvas.image, canvas.generated_mask)

    result_array = np.asarray(result)
    source_array = np.asarray(canvas.image)
    observed = np.asarray(canvas.observed_mask) > 127
    generated_area = np.asarray(canvas.generated_mask) > 127
    assert np.array_equal(result_array[observed], source_array[observed])
    assert np.all(result_array[generated_area] == np.array([200, 100, 50]))


def test_canvas_masks_are_complements() -> None:
    source = Image.new("RGB", (100, 80), "red")
    canvas = prepare_outpaint_canvas(
        source,
        Image.new("L", source.size, 255),
        aspect=CompletionAspect.PORTRAIT,
        margin_percent=70,
        target_long_edge=512,
    )
    combined = np.asarray(canvas.observed_mask, dtype=np.uint16) + np.asarray(
        canvas.generated_mask, dtype=np.uint16
    )
    assert np.all(combined == 255)
    assert canvas.image.width % 8 == 0
    assert canvas.image.height % 8 == 0


def test_variant_canvas_is_entirely_synthetic() -> None:
    canvas = prepare_variant_canvas(frame=VariantFrame.FULL_BODY, target_long_edge=768)
    assert canvas.observed_mask.getbbox() is None
    assert canvas.generated_mask.getextrema() == (255, 255)


def test_scan_images_respects_recursive_and_hidden_paths(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10)).save(tmp_path / "a.JPG")
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (10, 10)).save(nested / "b.webp")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    Image.new("RGB", (10, 10)).save(hidden / "c.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    assert [path.name for path in scan_images(tmp_path, recursive=False)] == ["a.JPG"]
    assert [path.name for path in scan_images(tmp_path, recursive=True)] == ["a.JPG", "b.webp"]


def test_load_rgb_and_atomic_writes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (20, 30), (255, 0, 0, 128)).save(source)
    loaded = load_rgb(source)
    assert loaded.mode == "RGB"
    assert loaded.size == (20, 30)

    output = tmp_path / "out.png"
    save_png(loaded, output, {"seed": 12})
    assert output.exists()
    with Image.open(output) as saved:
        metadata = saved.info["BodyBuilder"]
    assert json.loads(metadata)["seed"] == 12

    sidecar = tmp_path / "out.json"
    write_json(sidecar, {"value": "ok"})
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"value": "ok"}
