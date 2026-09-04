"""Typed application configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class BackendKind(StrEnum):
    SDXL = "sdxl"
    CLASSICAL = "classical"


class SubjectKind(StrEnum):
    AUTO = "auto"
    PERSON = "person"
    OBJECT = "object"


class DeviceKind(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class CompletionAspect(StrEnum):
    AUTO = "auto"
    PORTRAIT = "portrait"
    SQUARE = "square"
    LANDSCAPE = "landscape"


class VariantFrame(StrEnum):
    FULL_BODY = "full_body"
    PORTRAIT = "portrait"
    SQUARE = "square"
    LANDSCAPE = "landscape"


@dataclass(slots=True)
class ModelSettings:
    inpainting_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ip_adapter_model_id: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    upscaler_model_id: str = "caidas/swin2SR-classical-sr-x2-64"
    grouping_model_id: str = "facebook/dinov2-small"


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    subject_kind: SubjectKind = SubjectKind.AUTO
    backend: BackendKind = BackendKind.SDXL
    device: DeviceKind = DeviceKind.AUTO
    completion_aspect: CompletionAspect = CompletionAspect.AUTO
    variant_frame: VariantFrame = VariantFrame.FULL_BODY
    completion_margin_percent: int = 60
    completions_per_source: int = 1
    synthetic_variants: int = 0
    target_long_edge: int = 1024
    inference_steps: int = 30
    guidance_scale: float = 6.0
    denoising_strength: float = 0.98
    reference_fidelity: float = 0.75
    seed: int = 137
    group_all_images: bool = True
    grouping_similarity_threshold: float = 0.73
    stitch_overlaps: bool = True
    max_stitch_candidates: int = 6
    upscale_2x: bool = True
    recursive_scan: bool = True
    continue_on_error: bool = True
    custom_prompt: str = ""
    run_name: str = ""
    model_settings: ModelSettings = field(default_factory=ModelSettings)

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self._validate()

    def _validate(self) -> None:
        if not 0 <= self.completion_margin_percent <= 300:
            raise ValueError("Completion margin must be between 0 and 300 percent")
        if not 1 <= self.completions_per_source <= 8:
            raise ValueError("Completions per source must be between 1 and 8")
        if not 0 <= self.synthetic_variants <= 16:
            raise ValueError("Synthetic variants must be between 0 and 16")
        if self.target_long_edge < 512 or self.target_long_edge > 2048:
            raise ValueError("AI long edge must be between 512 and 2048 pixels")
        if self.target_long_edge % 8:
            raise ValueError("AI long edge must be divisible by 8")
        if not 1 <= self.inference_steps <= 150:
            raise ValueError("Inference steps must be between 1 and 150")
        if not 0 <= self.guidance_scale <= 30:
            raise ValueError("Text guidance must be between 0 and 30")
        if not 0.05 <= self.denoising_strength <= 1:
            raise ValueError("Denoising strength must be between 0.05 and 1")
        if not 0 <= self.reference_fidelity <= 1.5:
            raise ValueError("Reference fidelity must be between 0 and 1.5")
        if not 0 <= self.grouping_similarity_threshold <= 1:
            raise ValueError("Grouping similarity threshold must be between 0 and 1")
        if not 0 <= self.max_stitch_candidates <= 24:
            raise ValueError("Maximum stitch candidates must be between 0 and 24")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["input_dir"] = str(self.input_dir)
        result["output_dir"] = str(self.output_dir)
        return result
