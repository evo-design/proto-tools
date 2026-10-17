"""tests/masked_models_tests/test_sampling.py.

Integration tests for sampling tools (masked models + random mutagenesis).
"""

from __future__ import annotations

from typing import Any

import pytest

from proto_tools.tools.masked_models.esm2 import (
    ESM2SampleConfig,
    ESM2SampleInput,
    run_esm2_sample,
)
from proto_tools.tools.masked_models.esm3 import (
    ESM3SampleConfig,
    ESM3SampleInput,
    run_esm3_sample,
)
from proto_tools.tools.mutagenesis.random_nucleotide import (
    RandomNucleotideSampleConfig,
    RandomNucleotideSampleInput,
    run_random_nucleotide_sample,
)
from proto_tools.tools.mutagenesis.random_protein import (
    RandomProteinSampleConfig,
    RandomProteinSampleInput,
    run_random_protein_sample,
)
from proto_tools.transforms.masking import MaskingStrategy
from proto_tools.utils import ToolInstance
from proto_tools.utils.sequence import (
    return_invalid_dna_chars,
    return_invalid_protein_chars,
)
from tests.conftest import make_persistent_fixture

_persistent_esm2 = make_persistent_fixture("esm2")
_persistent_esm3 = make_persistent_fixture("esm3")

# Validators per entity type
_VALIDATORS = {
    "protein": return_invalid_protein_chars,
    "nucleotide": return_invalid_dna_chars,
}

# ── Protein sampling tools ────────────────────────────────────────────────────
# Each entry: (run_fn, input_class, config_class, entity_type, has_logits)
PROTEIN_SAMPLING_TOOLS = [
    pytest.param(
        run_esm2_sample,
        ESM2SampleInput,
        ESM2SampleConfig,
        "protein",
        True,
        id="esm2",
        marks=pytest.mark.uses_gpu,
    ),
    pytest.param(
        run_esm3_sample,
        ESM3SampleInput,
        ESM3SampleConfig,
        "protein",
        True,
        id="esm3",
        marks=pytest.mark.uses_gpu,
    ),
    pytest.param(
        run_random_protein_sample,
        RandomProteinSampleInput,
        RandomProteinSampleConfig,
        "protein",
        False,
        id="random-protein",
    ),
]

MASKED_PROTEIN_SEQUENCES = [
    pytest.param(["_K_AY_AK_R"], id="multiple_masks_first_position"),
    pytest.param(["__________"], id="all_masked"),
    pytest.param(["MK_A", "MKTAY_AKQR_VQ_"], id="variable_length_batch_last_position"),
]

# ── Nucleotide sampling tools ─────────────────────────────────────────────────

NUCLEOTIDE_SAMPLING_TOOLS = [
    pytest.param(
        run_random_nucleotide_sample,
        RandomNucleotideSampleInput,
        RandomNucleotideSampleConfig,
        "nucleotide",
        False,
        id="random-nucleotide",
    ),
]

MASKED_NUCLEOTIDE_SEQUENCES = [
    pytest.param(["_C_TA_CG_A"], id="multiple_masks_first_position"),
    pytest.param(["__________"], id="all_masked"),
    pytest.param(["AC_A", "ACGTA_ACGT_GC_"], id="variable_length_batch_last_position"),
]


# ── Shared validation ─────────────────────────────────────────────────────────


def _validate_sample_output(sequences, result_sequences, entity_type):
    """Shared assertions for all sampling tests."""
    assert len(result_sequences) == len(sequences)
    validator = _VALIDATORS[entity_type]
    for orig, sampled in zip(sequences, result_sequences, strict=False):
        # Length preserved
        assert len(sampled) == len(orig)
        # All masks filled
        assert "_" not in sampled
        # Non-masked positions unchanged
        for i, (o, s) in enumerate(zip(orig, sampled, strict=False)):
            if o != "_":
                assert s == o, f"Position {i}: expected '{o}', got '{s}'"
        # All characters are valid for this entity type
        invalid = validator(sampled)
        assert not invalid, f"Invalid characters in output: {invalid}"


# ── Mask-fill tests ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "run_fn, input_cls, config_cls, entity_type, has_logits",
    PROTEIN_SAMPLING_TOOLS,
)
@pytest.mark.parametrize("sequences", MASKED_PROTEIN_SEQUENCES)
def test_protein_sample_fills_masks(
    run_fn,
    input_cls,
    config_cls,
    entity_type,
    has_logits,
    sequences,
):
    """Sampling with pre-masked protein sequences.

    Verifies that for each (tool x mask pattern) combination:
    - Every '_' position is replaced with a sampled character
    - Non-masked positions are left unchanged (context preserved)
    - Output sequence length matches input
    - All sampled characters are valid amino acids
    - Logits shape is correct (for tools that return them)
    """
    inputs = input_cls(sequences=sequences)
    if has_logits:
        config = config_cls(return_logits=True)
        result = run_fn(inputs, config)
    else:
        result = run_fn(inputs)

    _validate_sample_output(sequences, result.sequences, entity_type)

    if has_logits:
        assert len(result.results) == len(sequences)
        for seq, sample in zip(sequences, result.results, strict=False):
            assert sample.logits is not None, "Logits should be returned"
            assert len(sample.logits) == len(seq), f"Expected {len(seq)} positions, got {len(sample.logits)}"
            assert len(sample.logits[0]) == 20, f"Expected vocab size 20, got {len(sample.logits[0])}"


ITERATIVE_PROTEIN_SAMPLING_TOOLS = [
    pytest.param(run_esm2_sample, ESM2SampleInput, ESM2SampleConfig, id="esm2", marks=pytest.mark.uses_gpu),
    pytest.param(run_esm3_sample, ESM3SampleInput, ESM3SampleConfig, id="esm3", marks=pytest.mark.uses_gpu),
]


@pytest.mark.parametrize("run_fn, input_cls, config_cls", ITERATIVE_PROTEIN_SAMPLING_TOOLS)
def test_protein_sample_iterative_refinement(run_fn, input_cls, config_cls):
    """Iterative refinement satisfies the same length/validity invariants as single_pass.

    Uses ``num_steps=3`` to keep GPU runtime modest; the algorithm doesn't
    converge but every '_' must still be filled with a valid amino acid, and
    the final-forward path must produce correctly-shaped per-position logits.
    """
    sequences = ["MK_A", "MKTAY_AKQR_VQL"]
    config = config_cls(sampling_method="iterative_refinement", num_steps=3, return_logits=True)
    result = run_fn(input_cls(sequences=sequences), config)
    _validate_sample_output(sequences, result.sequences, "protein")

    assert len(result.results) == len(sequences)
    for seq, sample in zip(sequences, result.results, strict=False):
        assert sample.logits is not None
        assert len(sample.logits) == len(seq), f"Expected {len(seq)} positions, got {len(sample.logits)}"
        assert len(sample.logits[0]) == 20, f"Expected vocab size 20, got {len(sample.logits[0])}"


@pytest.mark.parametrize(
    "run_fn, input_cls, config_cls, entity_type, has_logits",
    NUCLEOTIDE_SAMPLING_TOOLS,
)
@pytest.mark.parametrize("sequences", MASKED_NUCLEOTIDE_SEQUENCES)
def test_nucleotide_sample_fills_masks(
    run_fn,
    input_cls,
    config_cls,
    entity_type,
    has_logits,
    sequences,
):
    """Sampling with pre-masked nucleotide sequences.

    Same invariants as the protein test, but with DNA sequences.
    """
    inputs = input_cls(sequences=sequences)
    result = run_fn(inputs)
    _validate_sample_output(sequences, result.sequences, entity_type)


# ── Masking strategy integration ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "run_fn, input_cls, config_cls, entity_type, has_logits",
    PROTEIN_SAMPLING_TOOLS,
)
def test_protein_sample_masking_strategy_integration(
    run_fn,
    input_cls,
    config_cls,
    entity_type,
    has_logits,
):
    """End-to-end masking strategy -> sampling pipeline for protein tools.

    Passes plain sequences (no '_' tokens) with a MaskingStrategy on the
    config. Verifies that the preprocess hook applies the strategy to
    produce masked sequences, and the sampler fills those masks with valid
    characters.
    """
    sequences = ["MKTAYIAKQR"]
    inputs = input_cls(sequences=sequences)
    config = config_cls(masking_strategy=MaskingStrategy(num_mutations=3))
    result = run_fn(inputs, config)

    assert len(result.sequences) == 1
    assert len(result.sequences[0]) == 10
    assert "_" not in result.sequences[0]
    invalid = _VALIDATORS[entity_type](result.sequences[0])
    assert not invalid, f"Invalid characters in output: {invalid}"

    # At most 3 positions can differ (could be fewer if the sampler
    # produces the original character at a masked position)
    diffs = sum(a != b for a, b in zip(sequences[0], result.sequences[0], strict=False))
    assert diffs <= 3, f"Expected at most 3 mutations, got {diffs}"


@pytest.mark.parametrize(
    "run_fn, input_cls, config_cls, entity_type, has_logits",
    NUCLEOTIDE_SAMPLING_TOOLS,
)
def test_nucleotide_sample_masking_strategy_integration(
    run_fn,
    input_cls,
    config_cls,
    entity_type,
    has_logits,
):
    """End-to-end masking strategy -> sampling pipeline for nucleotide tools."""
    sequences = ["ACGTACGTAC"]
    inputs = input_cls(sequences=sequences)
    config = config_cls(masking_strategy=MaskingStrategy(num_mutations=3))
    result = run_fn(inputs, config)

    assert len(result.sequences) == 1
    assert len(result.sequences[0]) == 10
    assert "_" not in result.sequences[0]
    invalid = _VALIDATORS[entity_type](result.sequences[0])
    assert not invalid, f"Invalid characters in output: {invalid}"

    diffs = sum(a != b for a, b in zip(sequences[0], result.sequences[0], strict=False))
    assert diffs <= 3, f"Expected at most 3 mutations, got {diffs}"


# ── Identity (all positions fixed) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "run_fn, input_cls, config_cls, entity_type, has_logits",
    PROTEIN_SAMPLING_TOOLS + NUCLEOTIDE_SAMPLING_TOOLS,
)
def test_all_fixed_positions_is_identity(
    run_fn,
    input_cls,
    config_cls,
    entity_type,
    has_logits,
):
    """When every position is fixed, output must equal input."""
    seq = "MKTL" if entity_type == "protein" else "ACGT"
    fixed = list(range(1, len(seq) + 1))
    config = config_cls(
        masking_strategy=MaskingStrategy(mask_fraction=1.0, fixed_positions=fixed),
    )
    result = run_fn(input_cls(sequences=[seq]), config)
    assert result.sequences == [seq]


# ── Excluded amino acids ──────────────────────────────────────────────────────

# Fully masked, so every output residue is drawn from the model's distribution.
_EXCLUDED_AAS = ["C", "L", "W"]

_VH_SEQ = (
    "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYY"
    "GDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSS"
)


def _sample_esm(run_fn, input_cls, config_cls, excluded, sampling_method):
    result = run_fn(
        input_cls(sequences=["_" * 24]),
        config_cls(
            excluded_amino_acids=excluded,
            sampling_method=sampling_method,
            num_steps=3,
            temperature=2.0,
            seed=5,
        ),
    )
    return result.sequences


def _sample_ablang(excluded, sampling_method):
    if sampling_method != "single_pass":
        pytest.skip("AbLang has no iterative-refinement path")
    from proto_tools.entities.antibody import Antibody
    from proto_tools.tools.masked_models.ablang import (
        AbLangSampleConfig,
        AbLangSampleInput,
        run_ablang_sample,
    )

    # AbLang needs antibody context, so mask the CDR3 loop rather than the whole chain.
    heavy = _VH_SEQ[:98] + "_" * 12 + _VH_SEQ[110:]
    result = run_ablang_sample(
        AbLangSampleInput(antibodies=[Antibody(heavy_chain=heavy)]),
        AbLangSampleConfig(excluded_amino_acids=excluded, temperature=2.0, seed=5),
    )
    return [sample.sequence[98:110] for sample in result.results]


def _sample_mixed(toolkit, checkpoint, excluded, sampling_method):
    from tests.masked_models_tests._mixed_checks import call_model

    # Protein-modality masks only; the DNA alphabet is unaffected by the setting.
    result = call_model(
        toolkit,
        "sample",
        checkpoint,
        {"sequences": ["<+>" + "_" * 12], "mask_modalities": [dict.fromkeys(range(2, 14), "protein")]},
        excluded_amino_acids=excluded,
        sampling_method=sampling_method,
        num_steps=3,
        temperature=2.0,
        seed=5,
    )
    return [sample.sequence[1:] for sample in result.results]


EXCLUSION_SAMPLERS = [
    pytest.param(
        lambda excluded, method: _sample_esm(run_esm2_sample, ESM2SampleInput, ESM2SampleConfig, excluded, method),
        id="esm2",
        marks=pytest.mark.uses_gpu,
    ),
    pytest.param(
        lambda excluded, method: _sample_esm(run_esm3_sample, ESM3SampleInput, ESM3SampleConfig, excluded, method),
        id="esm3",
        marks=pytest.mark.uses_gpu,
    ),
    pytest.param(_sample_ablang, id="ablang", marks=pytest.mark.uses_gpu),
    pytest.param(
        lambda excluded, method: _sample_mixed("glm2", "tattabio/gLM2_150M", excluded, method),
        id="glm2",
        marks=pytest.mark.uses_gpu,
    ),
    pytest.param(
        lambda excluded, method: _sample_mixed("minerva", "gbrixi/minerva-mlm", excluded, method),
        id="minerva",
        marks=pytest.mark.uses_gpu,
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("sampling_method", ["single_pass", "iterative_refinement"])
@pytest.mark.parametrize("sample_fn", EXCLUSION_SAMPLERS)
def test_excluded_amino_acids_never_sampled(sample_fn, sampling_method):
    """Excluded residues never appear at masked positions, and the rest still fill.

    High temperature on a fully masked window makes an unconstrained sampler
    overwhelmingly likely to emit at least one excluded residue, so a clean
    run is evidence the mask applied rather than luck.
    """
    sampled = sample_fn(_EXCLUDED_AAS, sampling_method)

    assert sampled, "sampler returned no sequences"
    for sequence in sampled:
        assert "_" not in sequence
        assert not set(sequence) & set(_EXCLUDED_AAS), f"excluded residue in {sequence}"
        assert not return_invalid_protein_chars(sequence)


# ── Cross-toolkit overlay isolation ───────────────────────────────────────────


@pytest.mark.uses_gpu
def test_esm2_sample_entropy_masking_shares_esm2_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESM2 sampling with entropy masking scores via the same 'esm2' worker.

    Model-informed masking always uses the sampling tool's own model, so the
    preprocess scoring dispatch and the main sampling dispatch resolve to the
    single 'esm2' auto-persist overlay entry — no second toolkit is created.
    """
    from proto_tools.tools.masked_models.esm2 import esm2_sample as esm2_sample_mod
    from proto_tools.utils.tool_instance import _auto_persist_overlay

    snapshots: list[tuple[str, set[str]]] = []

    def _snap(where: str) -> None:
        overlay = _auto_persist_overlay.get() or {}
        snapshots.append((where, set(overlay.keys())))

    original_preprocess = ESM2SampleConfig.preprocess

    def _instrumented_preprocess(self: ESM2SampleConfig, inputs: Any) -> Any:
        _snap("preprocess-enter")
        result = original_preprocess(self, inputs)
        _snap("preprocess-exit")
        return result

    monkeypatch.setattr(ESM2SampleConfig, "preprocess", _instrumented_preprocess)

    # Patch the name as re-bound in esm2_sample's module namespace (it's imported
    # via `from proto_tools.transforms.masking import build_position_score_fn`).
    original_build = esm2_sample_mod.build_position_score_fn

    def _instrumented_build(
        sampling_model: str,
        masking_strategy: MaskingStrategy,
        device: str,
        sampling_checkpoint: str | None = None,
    ) -> Any:
        fn = original_build(sampling_model, masking_strategy, device, sampling_checkpoint=sampling_checkpoint)
        if fn is None:
            return None

        def _wrapped(sequences: list[str]) -> Any:
            _snap("build_position_score_fn-called")
            return fn(sequences)

        return _wrapped

    monkeypatch.setattr(esm2_sample_mod, "build_position_score_fn", _instrumented_build)

    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "GVKLIFYEASKILDHLKVAARKVPQRSSGFGIP",
    ]
    config = ESM2SampleConfig(
        masking_strategy=MaskingStrategy(method="entropy", num_mutations=3),
        temperature=1.0,
        batch_size=1,
        device="cuda",
    )
    # An isolated cache hides the module's persistent esm2 worker so the auto-persist overlay is seeded.
    with ToolInstance.scope():
        result = run_esm2_sample(ESM2SampleInput(sequences=sequences), config)

    assert result.success, f"run_esm2_sample failed: {result.errors}"
    assert len(result.sequences) == 2
    assert all(isinstance(s, str) and len(s) > 0 for s in result.sequences)

    preprocess_snaps = [s for s in snapshots if s[0] in ("preprocess-enter", "preprocess-exit")]
    assert preprocess_snaps, "preprocess was never entered"
    for where, keys in preprocess_snaps:
        assert "esm2" in keys, f"snapshot {where}: expected 'esm2' in overlay, got {keys}"

    score_snaps = [s for s in snapshots if s[0] == "build_position_score_fn-called"]
    assert score_snaps, "entropy scoring path never fired — test would be vacuous"
    for where, keys in score_snaps:
        # Scoring reuses the sampling toolkit's worker — only 'esm2', never a second toolkit.
        assert keys == {"esm2"}, f"snapshot {where}: expected only 'esm2' during scoring, got {keys}"
