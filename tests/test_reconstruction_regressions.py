"""Technical regressions; model calls are mocked and do not test visual fidelity."""

from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from bodybuilder.ai.base import BackendFatalError, GenerationBackend
from bodybuilder.ai.sdxl import SdxlBackend, prepare_reference_images
from bodybuilder.config import (
    CompletionAspect,
    DeviceKind,
    ModelSettings,
    PipelineConfig,
    SubjectKind,
)
from bodybuilder.core.canvas import prepare_outpaint_canvas, preserve_observed_pixels
from bodybuilder.core.image_io import fragment_mask_path, load_fragment, load_rgb, scan_images
from bodybuilder.core.pipeline import AnalysisRunResult, PipelineCallbacks, ReconstructionPipeline
from bodybuilder.core.types import GenerationRequest, ImageAnalysis
from bodybuilder.core.validation import (
    InvalidGenerationError,
    decode_model_output,
    mask_like,
    validate_generation,
)


def texture(size=(512, 512)):
    array = np.random.default_rng(9).integers(20, 230, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def request():
    canvas = Image.new("RGB", (512, 512), (127, 127, 127))
    reference = texture((40, 50))
    return GenerationRequest(canvas, Image.new("L", canvas.size, 255), reference, None,
        "Complete this photograph", "collage", 137, 40, 5, 1, 0.6, 512, 512,
        reference_images=(reference,))


@pytest.mark.parametrize("fill", ["white", "black", (127, 127, 127), (200, 40, 70)])
def test_blank_outputs_are_not_accepted(fill):
    data = request()
    with pytest.raises(InvalidGenerationError):
        validate_generation(Image.new("RGB", data.canvas.size, fill), data.canvas, data.generated_mask)


def test_nonfinite_pixels_are_rejected_before_conversion():
    with pytest.raises(InvalidGenerationError, match="non-finite"):
        decode_model_output(np.full((12, 12, 3), np.nan))


def test_binary_mask_is_not_a_photo_but_grayscale_photo_is():
    mask = Image.new("RGB", (64, 64), "white")
    mask.paste((0, 0, 0), (10, 10, 40, 40))
    assert mask_like(mask)
    assert not mask_like(texture().convert("L"))


def test_unchanged_missing_region_is_rejected():
    image = texture()
    with pytest.raises(InvalidGenerationError, match="unchanged"):
        validate_generation(image, image.copy(), Image.new("L", image.size, 255))


def test_transparency_is_missing_not_white_evidence(tmp_path):
    path = tmp_path / "fragment.png"
    image = Image.new("RGBA", (32, 24), (20, 60, 100, 255))
    image.paste((0, 0, 0, 0), (10, 0, 32, 24))
    image.save(path)
    pixels, observed = load_fragment(path)
    assert observed.getpixel((2, 2)) == 255
    assert observed.getpixel((15, 2)) == 0
    assert pixels.getpixel((15, 2)) == (127, 127, 127)


def test_translucent_images_can_be_previewed_but_not_claimed_as_observed(tmp_path):
    path = tmp_path / "translucent.png"
    Image.new("RGBA", (20, 30), (255, 0, 0, 128)).save(path)
    assert load_rgb(path).size == (20, 30)
    with pytest.raises(ValueError, match="No observed pixels"):
        load_fragment(path)


def test_explicit_hole_masks_and_original_file_preserved(tmp_path):
    path = tmp_path / "photo.jpg"
    texture((40, 30)).save(path)
    original = path.read_bytes()
    mask = Image.new("L", (40, 30), 0)
    mask.paste(255, (10, 5, 25, 20))
    mask.save(fragment_mask_path(path))
    image, observed = load_fragment(path)
    assert observed.getpixel((11, 6)) == 0
    assert image.getpixel((11, 6)) == (127, 127, 127)
    assert observed.getpixel((1, 1)) == 255
    assert path.read_bytes() == original
    assert scan_images(tmp_path) == [path]


def test_previous_runs_and_masks_not_scanned_as_sources(tmp_path):
    source = tmp_path / "photo.png"
    texture((12, 12)).save(source)
    prior = tmp_path / "previous" / "diagnostics"
    prior.mkdir(parents=True)
    (prior.parent / "run_manifest.json").write_text("{}")
    texture((12, 12)).save(prior / "reference.png")
    Image.new("L", (12, 12), 255).save(tmp_path / "something_generated_mask.png")
    assert scan_images(tmp_path) == [source]


def test_narrow_fragment_gets_viable_canvas_and_preserves_visible_pixels():
    image = texture((12, 1000))
    canvas = prepare_outpaint_canvas(image, Image.new("L", image.size, 255),
        aspect=CompletionAspect.AUTO, margin_percent=100, target_long_edge=1024)
    assert min(canvas.image.size) >= 512
    result = preserve_observed_pixels(texture(canvas.image.size), canvas.image, canvas.generated_mask)
    observed = np.asarray(canvas.observed_mask) > 127
    assert np.array_equal(np.asarray(result)[observed], np.asarray(canvas.image)[observed])


def test_references_keep_fragment_edges():
    image = Image.new("RGB", (500, 100), "blue")
    image.paste((255, 0, 0), (0, 0, 100, 100))
    padded = prepare_reference_images((image,))[0]
    assert padded.size == (224, 224)
    assert padded.getpixel((10, 112))[0] > 200


def backend():
    return SdxlBackend(subject_kind=SubjectKind.PERSON, device=DeviceKind.AUTO,
                       models=ModelSettings(), use_face_adapter=False)


def test_blank_generation_retried_once_then_validated(monkeypatch):
    model = backend()
    monkeypatch.setattr(model, "prepare", lambda: None)
    seeds = []
    def generate(data, seed, fidelity, cancelled, progress):
        seeds.append(seed)
        return Image.new("RGB", data.canvas.size, "white") if len(seeds) == 1 else texture()
    monkeypatch.setattr(model, "_generate_once", generate)
    assert model.generate(request(), cancel_event=threading.Event()).size == (512, 512)
    assert seeds == [137, 138]
    assert model.last_generation_metadata["effective_seed"] == 138


def test_repeated_blank_output_fails_instead_of_fake_success(monkeypatch):
    model = backend()
    monkeypatch.setattr(model, "prepare", lambda: None)
    monkeypatch.setattr(model, "_generate_once", lambda *args: Image.new("RGB", (512, 512), "black"))
    with pytest.raises(BackendFatalError, match="failed twice"):
        model.generate(request(), cancel_event=threading.Event())


def test_missing_references_are_fatal():
    data = request()
    data.reference_images = ()
    with pytest.raises(BackendFatalError, match="individual reference"):
        backend().generate(data, cancel_event=threading.Event())


def mock_stack(monkeypatch, *, adapter_fails=False, tiling_error=None, cuda_available=True):
    import sys
    calls = []
    class Generator:
        def __init__(self, **kwargs):
            pass
        def manual_seed(self, seed):
            return self
    torch = SimpleNamespace(float16="fp16", float32="fp32", Generator=Generator,
        cuda=SimpleNamespace(is_available=lambda: cuda_available, empty_cache=lambda: None),
        backends=SimpleNamespace(), inference_mode=nullcontext, isfinite=np.isfinite)
    def enable_tiling():
        calls.append(("tiling", {}))
        if tiling_error is not None:
            raise tiling_error
    class Pipe:
        # Deliberately no pipeline.enable_vae_tiling convenience wrapper: use the
        # VAE's real API shape so an invented mock method cannot conceal this bug.
        vae = SimpleNamespace(
            register_to_config=lambda **kwargs: calls.append(("vae", kwargs)),
            enable_tiling=enable_tiling,
        )
        scheduler = SimpleNamespace(config={})
        def load_ip_adapter(self, *args, **kwargs):
            calls.append(("adapter", kwargs))
            if adapter_fails:
                raise ValueError("wrong encoder")
        def enable_model_cpu_offload(self):
            calls.append(("offload", {}))
        def to(self, device):
            calls.append(("to", {"device": device}))
            return self
        def set_ip_adapter_scale(self, scale):
            pass
        def __call__(self, **kwargs):
            calls.append(("generate", kwargs))
            return SimpleNamespace(images=np.asarray(texture(), dtype=np.float32)[None] / 255)
    def pipe_factory(*args, **kwargs):
        calls.append(("pipe", kwargs))
        return Pipe()
    def encoder_factory(*args, **kwargs):
        calls.append(("encoder", kwargs))
        return "correct-encoder"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoPipelineForInpainting=SimpleNamespace(from_pretrained=pipe_factory),
        EulerDiscreteScheduler=SimpleNamespace(from_config=lambda config: SimpleNamespace(config=config))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPVisionModelWithProjection=SimpleNamespace(from_pretrained=encoder_factory)))
    return calls


def test_correct_encoder_mandatory_adapter_and_nested_individual_images(monkeypatch):
    calls = mock_stack(monkeypatch)
    model = backend()
    model.generate(request(), cancel_event=threading.Event())
    mapping = dict(calls)
    assert mapping["encoder"]["subfolder"] == "models/image_encoder"
    assert mapping["pipe"]["image_encoder"] == "correct-encoder"
    assert mapping["adapter"]["image_encoder_folder"] is None
    names = [name for name, _ in calls]
    assert names.index("adapter") < names.index("tiling") < names.index("offload")
    assert mapping["vae"]["force_upcast"] is True
    assert not hasattr(model._pipe, "enable_vae_tiling")
    assert len(mapping["generate"]["ip_adapter_image"]) == 1
    assert isinstance(mapping["generate"]["ip_adapter_image"][0], list)


def test_prepare_uses_vae_api_on_cpu_and_is_idempotent(monkeypatch):
    calls = mock_stack(monkeypatch, cuda_available=False)
    model = backend()
    model.prepare()
    pipe = model._pipe
    model.prepare()
    assert model._pipe is pipe
    assert dict(calls)["to"]["device"] == "cpu"
    names = [name for name, _ in calls]
    assert names.count("tiling") == 1
    assert names.count("pipe") == 1
    assert names.index("tiling") < names.index("to")
    assert "offload" not in names


@pytest.mark.parametrize("error_type", [AttributeError, RuntimeError])
def test_vae_initialization_error_cleans_up_and_keeps_cause(monkeypatch, error_type):
    original = error_type("VAE tiling setup failed")
    calls = mock_stack(monkeypatch, tiling_error=original)
    model = backend()
    with pytest.raises(BackendFatalError, match="VAE tiling setup failed") as caught:
        model.prepare()
    assert caught.value.__cause__ is original
    assert model._pipe is None
    assert "offload" not in dict(calls)
    assert "generate" not in dict(calls)


def test_failed_initialization_can_be_retried_after_fixing_stack(monkeypatch):
    calls = mock_stack(monkeypatch, tiling_error=AttributeError("incompatible API"))
    model = backend()
    with pytest.raises(BackendFatalError):
        model.prepare()
    assert "generate" not in dict(calls)
    retry_calls = mock_stack(monkeypatch)
    model.prepare()
    assert model._pipe is not None
    assert "tiling" in dict(retry_calls)
    assert "offload" in dict(retry_calls)


def test_adapter_failure_never_continues_without_references(monkeypatch):
    calls = mock_stack(monkeypatch, adapter_fails=True)
    model = backend()
    with pytest.raises(BackendFatalError, match="Stopped rather"):
        model.prepare()
    assert model._pipe is None
    assert "generate" not in dict(calls)


class FakeBackend(GenerationBackend):
    def generate(self, data, **kwargs):
        return texture(data.canvas.size)


def test_pipeline_exports_only_photos_to_images_and_records_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "face.v1.png"
    texture((80, 100)).save(path)
    analysis = ImageAnalysis(path, 80, 100, path.stat().st_size, 1, 1, 1, 1, 1)
    monkeypatch.setattr("bodybuilder.core.pipeline.analyze_input_folder",
        lambda *args, **kwargs: AnalysisRunResult([analysis], []))
    shown = []
    pipeline = ReconstructionPipeline(PipelineConfig(source, tmp_path / "output"),
                                     callbacks=PipelineCallbacks(preview=shown.append))
    monkeypatch.setattr(pipeline, "_backend_for", lambda *args, **kwargs: FakeBackend())
    monkeypatch.setattr(pipeline, "_prompts", lambda *args: ("complete", "collage"))
    result = pipeline.run()
    assert len(result.output_paths) == 1
    output = result.output_paths[0]
    assert output.parent.name == "images"
    assert "face.v1__01.png" in output.name
    assert list(output.parent.iterdir()) == [output]
    assert shown == [output]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["outputs"][0]["reference_files"] == [str(path)]
    assert len(manifest["sources"][0]["sha256"]) == 64
    assert (result.run_dir / "bodybuilder.log").is_file()


def test_failed_backend_records_failure_no_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "photo.png"
    texture((40, 50)).save(path)
    analysis = ImageAnalysis(path, 40, 50, 100, 1, 1, 1, 1, 1)
    monkeypatch.setattr("bodybuilder.core.pipeline.analyze_input_folder",
        lambda *args, **kwargs: AnalysisRunResult([analysis], []))
    pipeline = ReconstructionPipeline(PipelineConfig(source, tmp_path / "output"))
    def fail(*args, **kwargs):
        raise BackendFatalError("No model")
    monkeypatch.setattr(pipeline, "_backend_for", fail)
    with pytest.raises(BackendFatalError):
        pipeline.run()
    assert json.loads((pipeline.run_dir / "run_manifest.json").read_text())["status"] == "failed"
    assert not (pipeline.run_dir / "images").exists()


def test_upscale_keeps_observed_pixels_and_masks_out_of_results(tmp_path):
    from bodybuilder.ai.upscale import Upscaler
    class DestructiveUpscaler(Upscaler):
        name = "test"
        def upscale(self, image):
            return Image.new("RGB", (image.width * 2, image.height * 2), "magenta")
    pipeline = ReconstructionPipeline(PipelineConfig(tmp_path / "source", tmp_path / "output", upscale_2x=True))
    pipeline._upscaler = DestructiveUpscaler()
    source = texture((32, 32))
    mask = Image.new("L", source.size, 0)
    mask.paste(255, (8, 8, 24, 24))
    record = pipeline._save_output(image=source, source_canvas=source, observed_mask=mask,
        generated_mask=ImageOps.invert(mask), output_base=tmp_path / "images" / "photo", metadata={})
    with Image.open(record["image"]) as saved:
        chosen = np.asarray(mask.resize(saved.size, Image.Resampling.NEAREST)) > 127
        assert np.array_equal(np.asarray(saved)[chosen], np.asarray(source.resize(saved.size, Image.Resampling.LANCZOS))[chosen])
    assert "diagnostics" in record["generated_mask"]
