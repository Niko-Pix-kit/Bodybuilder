"""Generic object and complementary-detail regressions, without model downloads."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bodybuilder.ai.subject_vision import Region, parse_regions
from bodybuilder.core.fragment_registration import assemble_overlap, repair_from_overlaps
from bodybuilder.core.subject_evidence import (
    PartEvidence,
    SourceView,
    SubjectEvidence,
    build_evidence,
    check_whole_framing,
    choose_common_subject,
    full_subject_prompt,
    subject_key,
    visible_part,
)


def textured(size=(240, 240)):
    return Image.fromarray(np.random.default_rng(17).integers(20, 230, (size[1], size[0], 3), dtype=np.uint8))


def view(name="a.png", label="vase"):
    image = textured()
    return SourceView(Path(name), image, Image.new("L", image.size, 255), [Region(label, (10, 10, 230, 230))])


def test_common_subject_is_generic_not_a_face_only_rule():
    assert choose_common_subject([view(), view("b.png")]) == "vase"
    assert subject_key("clock face") == "clock face"
    assert subject_key("woman") == "person"


def test_common_subject_ambiguity_is_not_silently_resolved():
    views = [view(), view("b.png")]
    for source in views:
        source.detections.append(Region("bottle", (10, 10, 230, 230)))
    with pytest.raises(ValueError, match="Several common"):
        choose_common_subject(views)
    assert choose_common_subject(views, "vase") == "vase"


def test_invalid_or_masked_parts_are_not_reference_evidence():
    source = view()
    source.image.paste((255, 255, 255), (20, 20, 90, 90))
    region = Region("mouth", (20, 20, 90, 90))
    assert visible_part(source, region, "mouth") is None
    source.image = textured()
    source.observed.paste(0, (20, 20, 90, 90))
    assert visible_part(source, region, "mouth") is None


def test_part_evidence_is_taken_from_complementary_source_files():
    sources = [view("eyes.png", "person"), view("mouth.png", "person")]
    class Vision:
        def check_cancelled(self):
            pass
        def describe_regions(self, image):
            return []
        def locate(self, image, label):
            name = "eyes" if image is sources[0].image else "mouth"
            return [Region(name, (30, 30, 100, 80))] if label == name else []
    evidence = build_evidence(sources, Vision())
    parts = {part.name: part.source.name for part in evidence.parts}
    assert parts == {"eyes": "eyes.png", "mouth": "mouth.png"}


def test_generic_object_parts_use_the_same_selection_path():
    sources = [view("left.png", "bicycle"), view("right.png", "bicycle")]
    class Vision:
        def check_cancelled(self):
            pass
        def describe_regions(self, image):
            name = "wheel" if image is sources[0].image else "handlebar"
            return [Region(name, (20, 20, 100, 100))]
    evidence = build_evidence(sources, Vision())
    assert {p.name for p in evidence.parts} == {"wheel", "handlebar"}


def test_reference_budget_reserves_slots_for_parts():
    evidence = SubjectEvidence("vase", [view(str(i)) for i in range(20)])
    evidence.parts = [PartEvidence("crack", Path("rare.png"), (0, 0, 32, 32), textured((32, 32)), 1)]
    images, records = evidence.references(limit=4)
    assert len(images) == 4
    assert records[-1]["source"] == "rare.png"


def test_synthetic_prompt_requires_new_whole_composition():
    evidence = SubjectEvidence("person", [view()])
    evidence.parts = [PartEvidence("eyes", Path("eyes.png"), (0, 0, 32, 32), textured((32, 32)), 1)]
    prompt, negative = full_subject_prompt(evidence, 0)
    assert "both feet in frame" in prompt
    assert "new coherent composition" in prompt
    assert "sunglasses" in negative
    evidence.accessory_labels.add("woman wearing sunglasses")
    assert "sunglasses" not in full_subject_prompt(evidence, 0)[1]
    prompt, _ = full_subject_prompt(SubjectEvidence("bicycle", [view()]), 1)
    assert "one complete bicycle" in prompt
    assert "three-quarter view" in prompt


def test_subject_framing_gate_handles_clipping_and_multiple_instances():
    assert check_whole_framing([Region("vase", (25, 25, 175, 175))], (200, 200)) is None
    assert check_whole_framing([Region("vase", (0, 10, 180, 195))], (200, 200))
    assert check_whole_framing([], (200, 200))
    assert check_whole_framing([Region("vase", (20, 20, 120, 180)), Region("vase", (80, 20, 180, 180))], (200, 200))


def test_detector_parser_accepts_open_vocabulary_schema_and_rejects_bad_boxes():
    result = parse_regions({"bboxes_labels": ["vase", "bad", "bad2"],
                            "bboxes": [[-1, 1, 120, 99], [2, 2, 0, 0], [0, 0, float("nan"), 90]]}, (100, 100))
    assert result == [Region("vase", (0, 1, 100, 99))]


def test_real_feature_alignment_recovers_a_hole_without_changing_observed_pixels():
    scene = textured((400, 300))
    anchor = SourceView(Path("anchor.png"), scene.copy(), Image.new("L", scene.size, 255), [])
    anchor.observed.paste(0, (180, 80, 240, 160))
    anchor.image.paste((127, 127, 127), (180, 80, 240, 160))
    donor = SourceView(Path("donor.png"), scene, Image.new("L", scene.size, 255), [])
    repaired, mask, records = repair_from_overlaps(anchor, [donor])
    assert records and records[0]["pixels_recovered"] > 1000
    known = np.asarray(anchor.observed) > 127
    assert np.array_equal(np.asarray(repaired)[known], np.asarray(anchor.image)[known])
    assert mask.getpixel((210, 110)) == 255


def test_genuine_overlapping_crops_are_assembled_on_a_union_canvas():
    scene = textured((500, 300))
    a, b = scene.crop((0, 0, 340, 300)), scene.crop((160, 0, 500, 300))
    anchor = SourceView(Path("a.png"), a, Image.new("L", a.size, 255), [])
    donor = SourceView(Path("b.png"), b, Image.new("L", b.size, 255), [])
    result = assemble_overlap(anchor, donor)
    assert result is not None
    image, observed, record = result
    assert image.width > 490 and record["pixels_added"] > 20000
    assert observed.size == image.size


def test_different_photos_are_not_pasted_as_matching_evidence():
    a = view()
    b = view("b.png")
    b.image = Image.new("RGB", a.image.size, "red")
    assert assemble_overlap(a, b) is None
