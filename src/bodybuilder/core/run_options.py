"""Desktop batch planning and real, bounded generation-quality budgets.

Low-level PipelineConfig defaults remain compatible with existing scripts. The
application uses desktop_config and explicitly selects a workflow and preset.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from bodybuilder.config import PipelineConfig

DEFAULT_SYNTHETIC_VIEWS = 5
MAX_SYNTHETIC_VIEWS = 16
T = TypeVar("T")


class WorkflowMode(StrEnum):
    BOTH = "sources_and_synthetic"
    SOURCES = "sources_only"
    SYNTHETIC = "synthetic_only"


class QualityMode(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class QualityProfile:
    mode: QualityMode
    working_long_edge: int
    generation_steps: int
    detail_steps: int
    max_detail_passes: int
    # All modes retain the same reference budget and evidence analysis. Fast
    # reduces generation work, not the requirement to use original evidence.
    reference_limit: int = 16

    def apply(self, config: PipelineConfig) -> PipelineConfig:
        return replace(config, target_long_edge=self.working_long_edge,
                       inference_steps=self.generation_steps)

    def report(self) -> dict[str, str | int]:
        return {**asdict(self), "mode": self.mode.value}

    def description(self) -> str:
        return (f"{self.working_long_edge}px working image; {self.generation_steps} generation steps; "
                f"up to {self.max_detail_passes} detail passes at {self.detail_steps} steps. "
                "Original evidence and observed-pixel protection stay enabled.")


def quality_profile(mode: QualityMode | str) -> QualityProfile:
    mode = QualityMode(mode)
    if mode == QualityMode.FAST:
        return QualityProfile(mode, 768, 24, 16, 2)
    if mode == QualityMode.HIGH:
        # Keep the native-size canvas instead of increasing GPU memory pressure
        # or shrinking a larger synthetic image again inside the 2x upscaler.
        return QualityProfile(mode, 1024, 60, 44, 8)
    return QualityProfile(mode, 1024, 40, 28, 4)


def desktop_config(input_dir: Path, output_dir: Path, *,
                   quality: QualityMode = QualityMode.BALANCED) -> PipelineConfig:
    """The public desktop defaults: five total synthetic views and 2x output."""
    config = PipelineConfig(input_dir=input_dir, output_dir=output_dir,
                            synthetic_variants=DEFAULT_SYNTHETIC_VIEWS, upscale_2x=True)
    return quality_profile(quality).apply(config)


def plan_tasks(views: Sequence[T], workflow: WorkflowMode | str, synthetic_count: int,
               *, completions_per_source: int = 1) -> list[T | None]:
    """Restore sources first, then generate exactly the requested new-view count.

    None denotes a free composition, never a generated image reused as evidence.
    There is no implicit bonus synthetic view in any of the new workflows.
    """
    workflow = WorkflowMode(workflow)
    if not views:
        raise ValueError("At least one original source photograph is required")
    if isinstance(synthetic_count, bool) or not isinstance(synthetic_count, int):
        raise ValueError("Synthetic view count must be an integer")
    if not 0 <= synthetic_count <= MAX_SYNTHETIC_VIEWS:
        raise ValueError(f"Synthetic view count must be between 0 and {MAX_SYNTHETIC_VIEWS}")
    if (isinstance(completions_per_source, bool) or not isinstance(completions_per_source, int)
            or not 1 <= completions_per_source <= 8):
        raise ValueError("Completions per source must be an integer between 1 and 8")
    if workflow == WorkflowMode.SYNTHETIC and not synthetic_count:
        raise ValueError("Select at least one synthetic view, or choose a source-restoration workflow")
    tasks: list[T | None] = []
    if workflow != WorkflowMode.SYNTHETIC:
        tasks.extend(view for view in views for _ in range(completions_per_source))
    if workflow != WorkflowMode.SOURCES:
        tasks.extend([None] * synthetic_count)
    return tasks
