"""Geometry and mask tests using real Pillow arrays, no model simulation."""
import numpy as np
import pytest
from PIL import Image

from bodybuilder.core.framing import assess_framing, make_recovery_canvas


def test_edge_contact_is_advisory_and_names_the_edges():
    check = assess_framing([(10, 0, 90, 100)], (100, 100))
    assert check.status == "edge_contact"
    assert check.edges == ("top", "bottom")
    assert not check.passed
    assert not check.report()["geometry_verified"]


def test_duplicate_boxes_are_not_multiple_subjects():
    check = assess_framing([(20, 20, 80, 80), (21, 21, 79, 79)], (100, 100))
    assert check.passed
    assert len(check.boxes) == 1


def test_distinct_substantial_subjects_are_reported():
    check = assess_framing([(5, 10, 45, 90), (55, 10, 95, 90)], (100, 100))
    assert check.status == "multiple_subjects"


@pytest.mark.parametrize("boxes", [[], [(10, 10, 0, 0)], [(0, 0, float('nan'), 10)]])
def test_missing_or_invalid_detection_does_not_become_a_pass(boxes):
    assert assess_framing(boxes, (100, 100)).status == "not_localized"


def test_boxes_outside_image_are_clipped_for_edge_check():
    check = assess_framing([(-10, 10, 110, 90)], (100, 100))
    assert check.edges == ("left", "right")


@pytest.mark.parametrize("size", [(512, 512), (680, 1024), (1024, 680)])
def test_recovery_preserves_resampled_center_and_requires_generated_border(size):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    recovery = make_recovery_canvas(image)
    assert recovery.image.size == recovery.missing.size == size
    left, top, right, bottom = recovery.draft_box
    expected = image.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    assert np.array_equal(np.asarray(recovery.image.crop(recovery.draft_box)), np.asarray(expected))
    assert recovery.missing.crop(recovery.draft_box).getextrema() == (0, 0)
    assert recovery.missing.getpixel((0, 0)) == 255
    assert left > 0 and top > 0 and right < size[0] and bottom < size[1]
    assert 0.45 < np.asarray(recovery.missing).mean() / 255 < 0.55


@pytest.mark.parametrize("scale", [0, 1, float('nan')])
def test_invalid_recovery_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        make_recovery_canvas(Image.new("RGB", (64, 64)), scale=scale)
