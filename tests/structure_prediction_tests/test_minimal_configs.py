"""tests/structure_prediction_tests/test_minimal_configs.py.

``minimal()`` is the cheap config the seed, lifecycle, smoke and folding tests run predictors with.
It must stay valid on the MSA path, and must never cost more than the defaults it replaces.
"""

from __future__ import annotations

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec

# Checkpoint values that run single-sequence and cannot condition on an MSA. A minimal(use_msa=True)
# config must never land on one of these, or the MSA path would be tested on a model that ignores it.
_MSA_FREE_CHECKPOINTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("model_checkpoint", "esmfold2-fast"),
        ("model_name", "protenix_mini_esm_v0.5.0"),  # ESM2 embeddings in place of MSAs
        ("model_name", "protenix_mini_ism_v0.5.0"),  # ISM embeddings in place of MSAs
    }
)

# Fields whose value is a count of samples, steps or recycles: minimal() may lower them, never raise them.
_COST_FIELDS = (
    "num_diffn_samples",
    "num_diffn_timesteps",
    "num_trunk_recycles",
    "num_samples",
    "num_steps",
    "num_cycles",
    "recycling_steps",
    "sampling_steps",
    "diffusion_samples",
    "sampling_steps_affinity",
    "diffusion_samples_affinity",
    "num_recycles",
    "num_diffusion_samples",
    "n_recycles",
    "diffusion_batch_size",
    "num_sampling_steps",
)


def _msa_specs() -> list[ToolSpec]:
    return [spec for spec in ToolRegistry.list_all() if "use_msa" in spec.config_model.model_fields]


@pytest.mark.parametrize("spec", _msa_specs(), ids=lambda spec: spec.key)
def test_minimal_msa_config_uses_an_msa_model(spec: ToolSpec) -> None:
    """``minimal(use_msa=True)`` builds, keeps use_msa, and picks a checkpoint that reads MSAs."""
    config = spec.config_model.minimal(use_msa=True)
    assert config.use_msa is True
    for field, value in _MSA_FREE_CHECKPOINTS:
        assert getattr(config, field, None) != value, f"{spec.key}: minimal(use_msa=True) chose {field}={value!r}"


@pytest.mark.parametrize("spec", _msa_specs(), ids=lambda spec: spec.key)
def test_minimal_is_never_more_expensive_than_the_defaults(spec: ToolSpec) -> None:
    defaults, minimal = spec.config_model(), spec.config_model.minimal()
    for field in _COST_FIELDS:
        default, cheap = getattr(defaults, field, None), getattr(minimal, field, None)
        if isinstance(default, int) and isinstance(cheap, int):
            assert cheap <= default, f"{spec.key}: minimal() raises {field} from {default} to {cheap}"


def test_an_explicit_checkpoint_still_wins() -> None:
    from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import ESMFold2Config

    assert ESMFold2Config.minimal().model_checkpoint == "esmfold2-fast"
    assert ESMFold2Config.minimal(use_msa=True).model_checkpoint == "esmfold2"
    assert ESMFold2Config.minimal(use_msa=False, model_checkpoint="esmfold2").model_checkpoint == "esmfold2"
