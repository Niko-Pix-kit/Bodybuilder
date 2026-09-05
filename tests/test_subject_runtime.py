"""Real library contracts, plus an explicitly enabled detector-weight smoke test."""
import importlib.util
import os
import threading
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw


def require_ai():
    missing = [name for name in ("torch", "diffusers", "transformers") if importlib.util.find_spec(name) is None]
    if missing:
        if os.environ.get("BODYBUILDER_REQUIRE_AI_TESTS") == "1":
            pytest.fail("Missing required AI libraries: " + ", ".join(missing))
        pytest.skip("Optional AI dependencies not installed")


def test_real_mask_processor_shape_and_free_composition_api():
    require_ai()
    from bodybuilder.ai.evidence_sdxl import EvidenceRequest, reference_conditioning
    image = Image.new("RGB", (512, 512), "gray")
    mask = Image.new("L", image.size, 255)
    request = EvidenceRequest(image, mask, image, None, "object", "crop", 1, 20, 5, 1,
                              0.6, 512, 512, reference_images=(image, image), reference_masks=(mask, mask))
    scales = []
    pipe = SimpleNamespace(set_ip_adapter_scale=scales.append)
    kwargs = reference_conditioning(pipe, request, 0.6)
    assert kwargs["cross_attention_kwargs"]["ip_adapter_masks"][0].shape == (1, 2, 512, 512)
    assert scales[-1] == [[0.6, 0.6]]
    request.reference_masks = ()
    request.free_composition = True
    assert reference_conditioning(pipe, request, 0.6) == {}
    assert scales[-1] == {"up": {"block_0": [0.0, 0.6, 0.0]}}


def test_native_florence_classes_exist_without_remote_python():
    require_ai()
    from transformers import Florence2ForConditionalGeneration, Florence2Processor
    assert callable(Florence2ForConditionalGeneration.from_pretrained)
    assert callable(Florence2Processor.post_process_generation)


@pytest.mark.skipif(os.environ.get("BODYBUILDER_TEST_VISION_WEIGHTS") != "1", reason="Explicit real-weight smoke test")
def test_subject_detector_with_real_weights():
    require_ai()
    from bodybuilder.ai.subject_vision import SubjectVision
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 140, 300, 230), fill="blue")
    draw.ellipse((105, 205, 155, 255), fill="black")
    draw.ellipse((230, 205, 280, 255), fill="black")
    vision = SubjectVision(threading.Event())
    try:
        # A synthetic fixture exercises actual tokenization, inference and parsing.
        # This deliberately makes no claim about recognizing a real person.
        assert isinstance(vision.detect(image), list)
        assert isinstance(vision.locate(image, "car"), list)
        assert isinstance(vision.describe_regions(image), list)
    finally:
        vision.close()
