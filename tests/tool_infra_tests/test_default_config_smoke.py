"""tests/tool_infra_tests/test_default_config_smoke.py.

Run each toolkit's default config once, for the toolkits whose tests otherwise use ``minimal()``.

Most GPU tests run a cheap ``minimal()`` config: a smaller checkpoint, or fewer samples, steps and
recycles. That keeps the suite fast but stops exercising what a user gets by default. This file
runs the default config on the tool's example input, once per toolkit, and only for toolkits no
other GPU test already runs at defaults. The guard below fails if a toolkit's ``minimal()``
departs from its defaults and nobody has decided how its default stays covered.
"""

from __future__ import annotations

import importlib

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec

# Keys run here at their default config. Each one's own tests pin a smaller checkpoint or cheaper
# settings, and the only test that runs its defaults is a benchmark, which ``--all`` does not run.
_SMOKE_KEYS: dict[str, str] = {
    "alphafold3-prediction": "test_af3.py runs defaults only in its benchmark",
    "boltz2-prediction": "test_boltz2.py pins 50 sampling steps; defaults only in its benchmark",
    "chai1-prediction": "test_chai1.py runs defaults only in its benchmark",
    "esmfold2-prediction": "test_esmfold2.py's prediction tests are mocked; defaults only in its benchmark",
    "glm2-score": "test_glm2.py pins gLM2_150M; the default is gLM2_650M",
    "opendde-prediction": "test_opendde.py pins 50 steps; defaults only in its benchmark",
    "progen2-score": "test_progen2.py pins progen2-small; the default is progen2-large",
    "progen3-score": "test_progen3.py pins progen3-112m; the default is progen3-762m",
    "protenix-prediction": "test_protenix.py pins mini/tiny models and 50 steps; the default is base v1.0",
    "rf3-prediction": "test_rf3.py pins diffusion_batch_size=1; defaults only in its benchmark",
}

# Toolkits whose minimal() departs from the defaults, with the GPU test that already runs them at
# defaults (module, test function). A smoke case for these would repeat that test's model load.
_COVERED_ELSEWHERE: dict[str, tuple[str, str]] = {
    "esm2": ("tests.masked_models_tests.test_esm2", "test_esm2_forward_pass"),
    "evo2": ("tests.causal_models_tests.test_evo2", "test_evo2_sample_tool"),
    "iglm": ("tests.causal_models_tests.test_iglm", "test_iglm_infill_designs_feed_straight_into_scoring"),
    "mol1": ("tests.structure_prediction_tests.test_mol1_prediction", "test_mol1_prediction_default_sampling"),
    "rfd4": ("tests.structure_design_tests.test_rfd4", "test_rfd4_design_scaffolds_a_motif"),
}

# Toolkits whose defaults are deliberately not run, and why.
_NOT_RUN_AT_DEFAULTS: dict[str, str] = {
    "bindcraft": "defaults are a multi-hour design campaign; minimal() is the documented smoke budget",
    "freebindcraft": "same as bindcraft",
    "germinal": "defaults allow 10,000 trajectories; its end-to-end test runs the pipeline at a test budget",
    "spliceai": "minimal() only supplies the reference genome a default config needs from the caller",
    "spliceai2": "same as spliceai",
}

# The fields that make a config cost more or load a different model.
_COST_FIELDS = frozenset(
    {
        "num_diffn_samples", "num_diffn_timesteps", "num_trunk_recycles", "num_samples", "num_steps",
        "num_cycles", "recycling_steps", "sampling_steps", "diffusion_samples", "sampling_steps_affinity",
        "diffusion_samples_affinity", "num_recycles", "num_diffusion_samples", "n_recycles",
        "diffusion_batch_size", "num_sampling_steps", "num_loops", "num_diffusion_steps",
        "num_pairformer_cycles", "nsteps", "max_trajectories", "soft_iterations", "temporary_iterations",
        "hard_iterations", "greedy_iterations", "num_seqs", "max_mpnn_sequences", "reference_fasta",
    }
)  # fmt: skip
_MODEL_FIELD_TOKENS = ("checkpoint", "model_name", "model_choice", "model_type", "model_variant", "weights_variant")


def _departs_from_defaults(spec: ToolSpec) -> bool:
    """Whether ``minimal()`` picks a different model or a cheaper setting than the defaults."""
    model = spec.config_model
    fields = {f for f in model.model_fields if f in _COST_FIELDS or any(t in f for t in _MODEL_FIELD_TOKENS)}
    try:
        defaults, minimal = model(), model.minimal()
    except Exception:
        return False  # a config that needs caller input (a local database, say) has no default run
    return any(getattr(defaults, f) != getattr(minimal, f) for f in fields)


def _gpu_toolkits_with_cheaper_minimal() -> set[str]:
    return {
        spec.source_file.parent.name
        for spec in ToolRegistry.list_all()
        if spec.uses_gpu and spec.category != "testing" and _departs_from_defaults(spec)
    }


# ── The guard (CPU) ───────────────────────────────────────────────────────────


def test_every_cheaper_minimal_has_a_decided_default_coverage() -> None:
    """A toolkit whose minimal() departs from its defaults is smoked here, covered elsewhere, or excused."""
    smoked = {ToolRegistry.get(key).source_file.parent.name for key in _SMOKE_KEYS}
    decided = smoked | set(_COVERED_ELSEWHERE) | set(_NOT_RUN_AT_DEFAULTS)
    undecided = sorted(_gpu_toolkits_with_cheaper_minimal() - decided)
    assert not undecided, (
        f"minimal() departs from the defaults for {undecided}, and nothing runs their defaults: add a key to "
        "_SMOKE_KEYS, name the test in _COVERED_ELSEWHERE, or give a reason in _NOT_RUN_AT_DEFAULTS"
    )
    assert not smoked & set(_COVERED_ELSEWHERE), "a smoke case would repeat a test that already runs defaults"


@pytest.mark.parametrize("toolkit", sorted(_COVERED_ELSEWHERE))
def test_the_named_default_coverage_exists(toolkit: str) -> None:
    module, function = _COVERED_ELSEWHERE[toolkit]
    assert callable(getattr(importlib.import_module(module), function, None)), f"{module}.{function} is gone"


# ── The smoke runs (GPU) ──────────────────────────────────────────────────────

# Each case runs one toolkit's default model; the session pool keeps it warm for that toolkit's
# neighbouring tests (tests/_warm_workers.py).
WARM_WORKERS = True


# Slow but not extensive: this is the only place these defaults run, so a plain --all run keeps it.
@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(_SMOKE_KEYS))
def test_default_config_runs(key: str) -> None:
    """The default model and settings run end to end.

    MSA generation stays off: it queries a remote MSA server, which is the network's behaviour rather
    than the model's, and test_structure_prediction.py's MSA cases already cover it.
    """
    spec = ToolRegistry.get(key)
    overrides = {"use_msa": False} if "use_msa" in spec.config_model.model_fields else {}
    result = spec.function(spec.example_input(), spec.config_model(**overrides))
    assert result.success, f"{key} failed at its default config: {result.errors}"
