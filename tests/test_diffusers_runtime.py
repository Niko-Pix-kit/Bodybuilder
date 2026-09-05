"""Real Diffusers API checks with tiny random components, without model downloads.

These tests verify initialization and VAE execution, not photographic fidelity.
Only pretrained weight loading and IP-Adapter weight loading are replaced.
The inpainting pipeline, scheduler, VAE, encoder and device placement are real.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from bodybuilder.ai.sdxl import SdxlBackend
from bodybuilder.config import DeviceKind, ModelSettings, SubjectKind

# Core-only installs may skip these tests. The AI compatibility CI job makes
# them mandatory, so a missing or incompatible dependency fails rather than skips.
_stack_missing = any(
    importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "transformers")
)
pytestmark = pytest.mark.skipif(
    _stack_missing and os.environ.get("BODYBUILDER_REQUIRE_AI_TESTS") != "1",
    reason="Real AI library checks require the optional AI dependencies",
)


@pytest.mark.parametrize("full_precision_retry", [False, True])
def test_prepare_with_real_inpaint_pipeline_and_tiled_vae(monkeypatch, full_precision_retry):
    import torch
    from diffusers import (
        AutoencoderKL,
        AutoPipelineForInpainting,
        EulerDiscreteScheduler,
        StableDiffusionXLInpaintPipeline,
    )
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    vae = AutoencoderKL(
        block_out_channels=(32,), norm_num_groups=8, sample_size=16,
        latent_channels=4, force_upcast=False,
    )
    encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=32, patch_size=16, projection_dim=32,
    ))
    pipe = StableDiffusionXLInpaintPipeline(
        vae=vae, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None, unet=None,
        scheduler=EulerDiscreteScheduler(), image_encoder=encoder,
        add_watermarker=False,
    )
    calls = []

    def load_encoder(*args, **kwargs):
        calls.append("encoder")
        assert kwargs["subfolder"] == "models/image_encoder"
        return encoder

    def load_pipeline(*args, **kwargs):
        calls.append("pipeline")
        assert kwargs["image_encoder"] is encoder
        return pipe

    def load_adapter(*args, **kwargs):
        calls.append("adapter")
        assert kwargs["image_encoder_folder"] is None

    monkeypatch.setattr(CLIPVisionModelWithProjection, "from_pretrained", load_encoder)
    monkeypatch.setattr(AutoPipelineForInpainting, "from_pretrained", load_pipeline)
    monkeypatch.setattr(pipe, "load_ip_adapter", load_adapter)
    model = SdxlBackend(
        subject_kind=SubjectKind.PERSON, device=DeviceKind.CPU,
        models=ModelSettings(), use_face_adapter=False,
    )
    model._full_precision = full_precision_retry
    try:
        assert not vae.use_tiling
        model.prepare()
        assert model._pipe is pipe
        assert vae.use_tiling
        assert vae.config.force_upcast
        assert next(vae.parameters()).device.type == "cpu"
        assert next(vae.parameters()).dtype == torch.float32
        assert isinstance(pipe.scheduler, EulerDiscreteScheduler)
        model.prepare()
        assert calls == ["encoder", "pipeline", "adapter"]

        # Exercise the actual tiled encoder and decoder, not just the API name.
        # The random weights do not represent the production SDXL checkpoint.
        with torch.inference_mode():
            sample = torch.zeros(1, 3, 24, 24)
            latents = vae.encode(sample).latent_dist.mode()
            decoded = vae.decode(latents).sample
        assert decoded.shape == sample.shape
        assert torch.isfinite(decoded).all()
    finally:
        model.close()
    assert model._pipe is None
