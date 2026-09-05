"""Batch counts and quality budgets are deterministic and independent of models."""
from dataclasses import FrozenInstanceError

import pytest

from bodybuilder.config import PipelineConfig
from bodybuilder.core.run_options import (
    QualityMode,
    WorkflowMode,
    desktop_config,
    plan_tasks,
    quality_profile,
)


def test_desktop_defaults_restore_sources_with_five_total_new_views(tmp_path):
    config = desktop_config(tmp_path / "source", tmp_path / "output")
    assert config.synthetic_variants == 5
    assert config.upscale_2x
    assert config.target_long_edge == 1024
    assert config.inference_steps == 40
    sources = [object(), object(), object()]
    tasks = plan_tasks(sources, WorkflowMode.BOTH, config.synthetic_variants)
    assert tasks[:3] == sources
    assert len(tasks) == 8
    assert sum(item is None for item in tasks) == 5


@pytest.mark.parametrize("count", [1, 5, 10, 16])
@pytest.mark.parametrize("workflow", list(WorkflowMode))
def test_counts_have_no_implicit_bonus_view(count, workflow):
    sources = ["a", "b"]
    tasks = plan_tasks(sources, workflow, count)
    assert tasks.count(None) == (0 if workflow == WorkflowMode.SOURCES else count)
    assert len([task for task in tasks if task is not None]) == (0 if workflow == WorkflowMode.SYNTHETIC else 2)


def test_source_copies_and_no_synthetic_work():
    assert plan_tasks(["a", "b"], WorkflowMode.BOTH, 0, completions_per_source=2) == ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="at least one synthetic"):
        plan_tasks(["a"], WorkflowMode.SYNTHETIC, 0)
    with pytest.raises(ValueError, match="original source"):
        plan_tasks([], WorkflowMode.BOTH, 5)


@pytest.mark.parametrize("count", [-1, 17, 1.5, True])
def test_invalid_counts_rejected(count):
    with pytest.raises(ValueError):
        plan_tasks(["a"], WorkflowMode.BOTH, count)


@pytest.mark.parametrize("mode,edge,steps,detail_steps,passes", [
    (QualityMode.FAST, 768, 24, 16, 2),
    (QualityMode.BALANCED, 1024, 40, 28, 4),
    (QualityMode.HIGH, 1024, 60, 44, 8),
])
def test_profiles_change_real_budgets_without_weakening_evidence(tmp_path, mode, edge, steps, detail_steps, passes):
    original = PipelineConfig(tmp_path, tmp_path / "out", synthetic_variants=10,
                              upscale_2x=True, reference_fidelity=0.7)
    profile = quality_profile(mode)
    configured = profile.apply(original)
    assert (configured.target_long_edge, configured.inference_steps) == (edge, steps)
    assert (profile.detail_steps, profile.max_detail_passes) == (detail_steps, passes)
    assert profile.reference_limit == 16
    assert configured.synthetic_variants == 10
    assert configured.upscale_2x and configured.reference_fidelity == 0.7
    assert original.target_long_edge == 1024 and original.inference_steps == 40
    assert configured is not original
    assert profile.report()["mode"] == mode.value
    with pytest.raises(FrozenInstanceError):
        profile.generation_steps = 1
