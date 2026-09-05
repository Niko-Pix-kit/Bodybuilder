"""Real Diffusers APIs with tiny random components, without model downloads.

These tests verify initialization, VAE execution and regional IP-Adapter routing,
not photographic fidelity. Pretrained weight loading alone is replaced.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from bodybuilder.ai.sdxl import SdxlBackend
from bodybuilder.config import DeviceKind, ModelSettings, SubjectKind

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
        with torch.inference_mode():
            sample = torch.zeros(1, 3, 24, 24)
            latents = vae.encode(sample).latent_dist.mode()
            decoded = vae.decode(latents).sample
        assert decoded.shape == sample.shape
        assert torch.isfinite(decoded).all()
    finally:
        model.close()
    assert model._pipe is None


def test_real_ip_adapter_routes_distinct_references_to_distinct_regions():
    from types import SimpleNamespace

    import torch
    from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0
    from PIL import Image

    from bodybuilder.ai.sdxl import reference_mask_array

    top = Image.new("L", (32, 32), 0)
    top.paste(255, (0, 0, 32, 16))
    bottom = Image.eval(top, lambda value: 255 - value)
    request = SimpleNamespace(reference_masks=(top, bottom), reference_images=(top, bottom), width=32, height=32)
    masks = [torch.from_numpy(reference_mask_array(request))]
    torch.manual_seed(8)
    processor = IPAdapterAttnProcessor2_0(hidden_size=16, cross_attention_dim=16,
                                         num_tokens=(4,), scale=[[.8, .8]])
    attention = Attention(query_dim=16, cross_attention_dim=16, heads=2, dim_head=8,
                          processor=processor).eval()
    hidden = torch.randn(2, 4, 16)
    text = torch.randn(2, 3, 16)
    references = torch.randn(2, 2, 4, 16)
    with torch.inference_mode():
        before = attention(hidden, encoder_hidden_states=(text, [references]), ip_adapter_masks=masks)
        changed = references.clone()
        changed[:, 0] += 2  # Change only the eye reference (top half).
        after = attention(hidden, encoder_hidden_states=(text, [changed]), ip_adapter_masks=masks)
    assert not torch.allclose(before[:, :2], after[:, :2])
    torch.testing.assert_close(before[:, 2:], after[:, 2:], rtol=0, atol=1e-6)
    assert torch.isfinite(after).all()
