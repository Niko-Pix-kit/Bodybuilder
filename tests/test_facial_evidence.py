"""Regional routing tests use synthetic patterns, not private user photographs."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from bodybuilder.ai.base import GenerationCancelled
from bodybuilder.ai.details import generate_with_details
from bodybuilder.ai.sdxl import reference_mask_array
from bodybuilder.core.evidence import (
    Guide,
    Region,
    collect_details,
    read_guide,
    save_guide,
    target_face,
    targeted_masks,
    valid_box,
)
from bodybuilder.core.image_io import fragment_mask_path, save_png, scan_images
from bodybuilder.core.types import GenerationRequest


def texture(size=(128, 160), seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(20, 240, (*size[::-1], 3), dtype=np.uint8))


def originals(tmp_path):
    upper, lower = tmp_path / "upper.png", tmp_path / "lower.png"
    texture().save(upper)
    texture(seed=1).save(lower)
    save_guide(upper, Guide([Region("eyes", (20, 15, 110, 55))], auto_eyes=False))
    save_guide(lower, Guide([Region("mouth", (35, 60, 95, 110))], auto_eyes=False))
    return upper, lower


def test_eye_and_mouth_can_come_from_different_originals(tmp_path):
    upper, lower = originals(tmp_path)
    chosen = collect_details([upper, lower])
    assert [(p.role, p.path) for p in chosen] == [("eyes", upper), ("mouth", lower)]
    assert all(len(p.source_sha256) == 64 for p in chosen)


def test_guide_preserves_original_and_invalidates_changed_source(tmp_path):
    path = tmp_path / "image.png"
    texture().save(path)
    before = path.read_bytes()
    save_guide(path, Guide([Region("mouth", (30, 60, 100, 105))], auto_eyes=False))
    assert path.read_bytes() == before
    assert read_guide(path, (128, 160)).regions[0].role == "mouth"
    texture(seed=3).save(path)
    with pytest.raises(ValueError, match="Source changed"):
        read_guide(path, (128, 160))


def test_masked_evidence_is_not_used_and_blank_patch_is_not_evidence(tmp_path):
    upper, lower = originals(tmp_path)
    mask = Image.new("L", (128, 160), 0)
    mask.paste(255, (0, 0, 128, 58))
    mask.save(fragment_mask_path(upper))
    assert [p.role for p in collect_details([upper, lower])] == ["mouth"]
    with pytest.raises(ValueError, match="visible"):
        save_guide(upper, Guide([Region("eyes", (20, 15, 110, 55))]))
    blank = tmp_path / "blank.png"
    Image.new("RGB", (128, 160), "white").save(blank)
    save_guide(blank, Guide([Region("mouth", (20, 20, 100, 90))], auto_eyes=False))
    assert collect_details([blank]) == ()


def test_copied_generated_png_cannot_contaminate_reference_set(tmp_path):
    upper, lower = originals(tmp_path)
    copied = tmp_path / "copied_output.png"
    save_png(texture(), copied, {"kind": "source_completion"})
    assert scan_images(tmp_path) == sorted([upper, lower])
    assert all(p.path != copied for p in collect_details([upper, copied, lower]))


@pytest.mark.parametrize("box", [(0, 0, 1, 20), (0, 0, float("nan"), 40), (-1, 0, 20, 30)])
def test_invalid_visible_boxes_rejected(box):
    with pytest.raises(ValueError):
        valid_box(box, (128, 160))


def test_face_location_can_extend_outside_the_crop(tmp_path):
    upper, lower = originals(tmp_path)
    guide = read_guide(lower, (128, 160))
    guide.face_box = (-10, -80, 140, 140)
    save_guide(lower, guide)
    face, method = target_face(lower, {"x": 100, "y": 150, "source_width": 64, "source_height": 80}, texture((256, 320)))
    assert face == (95, 110, 170, 220)
    assert method == "user_face_location"


def test_target_masks_never_include_observed_pixels(tmp_path):
    chosen = collect_details(originals(tmp_path))
    missing = Image.new("L", (128, 160), 255)
    missing.paste(0, (0, 0, 65, 160))
    kept, masks, union = targeted_masks(chosen, (10, 5, 120, 150), missing)
    assert len(kept) == 2
    assert masks[0].getbbox()[3] < masks[1].getbbox()[3]
    assert np.all(np.asarray(union)[:, :65] == 0)
    assert not np.array_equal(np.asarray(masks[0]), np.asarray(masks[1]))


def test_mask_array_matches_single_adapter_multiple_images_contract():
    request = SimpleNamespace(width=128, height=160, reference_images=(texture(), texture()),
                              reference_masks=(Image.new("L", (128, 160), 0), Image.new("L", (128, 160), 255)))
    array = reference_mask_array(request)
    assert array.shape == (1, 2, 160, 128)
    assert array.dtype == np.float32
    assert array[0, 0].max() == 0 and array[0, 1].min() == 1
    request.reference_masks = request.reference_masks[:1]
    with pytest.raises(ValueError, match="exactly one"):
        reference_mask_array(request)


def test_second_pass_routes_parts_and_keeps_all_observed_pixels(tmp_path, monkeypatch):
    upper, lower = originals(tmp_path)
    observed = Image.new("L", (128, 160), 0)
    observed.paste(255, (0, 130, 128, 160))
    canvas = texture()
    request = GenerationRequest(canvas, ImageOps.invert(observed), canvas, None,
        "photograph", "collage", 42, 30, 5, 1, .6, 128, 160,
        reference_images=(canvas,), evidence_paths=(upper, lower))
    calls = []
    class Engine:
        last_generation_metadata = {}
        log = staticmethod(lambda message: None)
        def _generate_validated(self, data, **kwargs):
            calls.append(data)
            return texture(seed=len(calls))
    monkeypatch.setattr("bodybuilder.ai.details.target_face", lambda *args: ((10, 5, 120, 150), "test_layout"))
    engine = Engine()
    result = generate_with_details(engine, request, cancel_event=threading.Event())
    assert len(calls) == 2
    assert "sunglasses" in calls[0].negative_prompt
    assert len(calls[1].reference_images) == len(calls[1].reference_masks) == 2
    assert not np.any(np.asarray(calls[1].generated_mask)[130:])
    assert np.array_equal(np.asarray(result)[130:], np.asarray(canvas)[130:])
    report = engine.last_generation_metadata["facial_guidance"]
    assert report["status"] == "regional_refinement_applied"
    assert report["identity_verified"] is False
    # References in the detail pass are original crops, not generated draft pixels.
    assert np.array_equal(np.asarray(calls[1].reference_images[0]), np.asarray(canvas.crop((20, 15, 110, 55))))


def test_uncertain_layout_is_reported_not_claimed_as_guided(tmp_path, monkeypatch):
    paths = originals(tmp_path)
    image = texture()
    data = GenerationRequest(image, Image.new("L", image.size, 255), image, None,
        "photo", "collage", 1, 30, 5, 1, .6, *image.size, reference_images=(image,), evidence_paths=paths)
    engine = SimpleNamespace(last_generation_metadata={}, log=lambda message: None,
                             _generate_validated=lambda *args, **kwargs: texture(seed=2))
    monkeypatch.setattr("bodybuilder.ai.details.target_face", lambda *args: (None, "face_not_located"))
    generate_with_details(engine, data, cancel_event=threading.Event())
    assert engine.last_generation_metadata["facial_guidance"]["status"] == "face_location_needed"
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GenerationCancelled):
        generate_with_details(engine, replace(data), cancel_event=cancelled)
