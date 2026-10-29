"""Tests for Evo 2 in-context DNA design."""

import pytest
from pydantic import ValidationError

import proto_tools.tools.causal_models.evo2.evo2_icl as evo2_icl
from proto_tools.tools.causal_models.evo2.evo2_icl import Evo2ICLConfig, Evo2ICLInput, run_evo2_icl
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from tests.conftest import benchmark_twice

_EXAMPLES = ["ACGTACGT", "TGCATGCA"]
_TWO_HUNDRED_BP_EXAMPLES = ["ACGT" * 50, "TGCA" * 50] * 8


@pytest.fixture(autouse=True)
def fresh_cache():
    """Prevent seeded mock outputs from leaking between tests."""
    token = _program_tool_cache.set(ToolCache())
    yield
    _program_tool_cache.reset(token)


@pytest.fixture
def mock_sampler(monkeypatch):
    """Capture standalone dispatch and allow tests to supply sampled continuations."""
    calls = []
    sampled = []

    def dispatch(toolkit, payload, *, instance=None, config=None):
        calls.append((toolkit, payload, instance, config))
        return {"sequences": list(sampled)}

    monkeypatch.setattr(evo2_icl.ToolInstance, "dispatch", staticmethod(dispatch))
    return calls, sampled


# ── Prompt construction and limits ────────────────────────────────────────────


@pytest.mark.parametrize("container", [list, tuple])
def test_evo2_icl_input_normalizes_preserving_order_and_multiplicity(container):
    """Whitespace and casing normalize without changing the demonstration sequence."""
    inputs = Evo2ICLInput(examples=container([" ac\ngt\t", "tgca", "AC GT"]))

    assert inputs.examples == ["ACGT", "TGCA", "ACGT"]
    assert Evo2ICLConfig(fixed_examples=True).compose_prompts(inputs)[0] == ["NACGTNTGCANACGTN"]


@pytest.mark.parametrize(
    ("examples", "prompt"),
    [("acgt", "NACGTN"), (["ACGT", "TGCA"], "NACGTNTGCAN")],
)
def test_evo2_icl_input_prompt_uses_one_n_per_boundary(examples, prompt):
    """Single and multiple examples share the same flanked, single-N format."""
    assert Evo2ICLConfig(fixed_examples=True).compose_prompts(Evo2ICLInput(examples=examples))[0] == [prompt]


@pytest.mark.parametrize("invalid", ["", " \n", "ACGN", "ACGU", ">example\nACGT"])
def test_evo2_icl_input_rejects_invalid_examples(invalid):
    """Examples require DNA bases, with N reserved for the assembled separator."""
    with pytest.raises(ValidationError, match="Example 2 must"):
        Evo2ICLInput(examples=["ACGT", invalid])


def test_evo2_icl_prompt_has_no_artificial_size_cap():
    """Large examples and separators compose fully rather than hitting a web quota."""
    config = Evo2ICLConfig(n_shots=2, num_sequences=1, fixed_examples=True, spacer="NN")
    inputs = Evo2ICLInput(examples=["A" * 50_000, "C" * 50_000])
    assert config.compose_prompts(inputs)[0] == ["NN" + "A" * 50_000 + "NN" + "C" * 50_000 + "NN"]
    assert Evo2ICLConfig(force_prompt_threshold=65_536).prefill_threshold == 65_536


def test_evo2_icl_local_counts_are_not_limited_by_web_quotas():
    """Large local requests compose normally beyond all former web-work quotas."""
    config = Evo2ICLConfig(num_sequences=5001, n_prompts=2, sequence_length=1001)
    prompts, _ = config.compose_prompts(Evo2ICLInput(examples=["A" * 32_766]))
    assert config.total_sequences == 10002
    assert len(prompts) == 2
    assert all(len(prompt) == 32_768 for prompt in prompts)


# ── Standalone sampling and output handling ────────────────────────────────────


def test_evo2_icl_dispatches_one_joint_prompt_for_each_design(mock_sampler):
    """FASTA examples jointly condition all designs rather than each producing a run."""
    calls, sampled = mock_sampler
    sampled.extend(["AAAA", "CCCC", "GGGG", "TTTT"])
    inputs = Evo2ICLInput(examples=["ACGT", "TGCA", "ACGT"])

    result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=4, sequence_length=4, fixed_examples=True, device="cuda"))

    assert len(calls) == 1
    assert calls[0][1]["prompts"] == ["NACGTNTGCANACGTN"] * 4
    assert result.sequences == sampled
    assert result.metadata["num_examples"] == 3
    assert result.metadata["prompt_length"] == len("N" + "N".join(inputs.examples) + "N")
    assert result.metadata["num_generated"] == 4
    assert result.tool_id == "evo2-icl"


@pytest.mark.parametrize("seed", [None, 63])
def test_evo2_icl_dispatch_propagates_sampling_controls_and_seed(mock_sampler, seed):
    """The new operation uses the existing sampler with the supplied decoding controls."""
    calls, sampled = mock_sampler
    sampled.extend(["AAAA", "CCCC"])
    inputs = Evo2ICLInput(examples=_EXAMPLES)
    config = Evo2ICLConfig(
        num_sequences=2,
        sequence_length=4,
        fixed_examples=True,
        temperature=0.7,
        top_k=2,
        top_p=0.8,
        batch_size=2,
        device="cuda",
        verbose=False,
        timeout=900,
        seed=seed,
    )

    result = run_evo2_icl(inputs, config)

    toolkit, payload, instance, dispatched_config = calls[0]
    assert toolkit == "evo2"
    assert instance is None
    assert dispatched_config.timeout == 900
    assert payload == {
        "operation": "sample",
        "prompts": ["NACGTACGTNTGCATGCAN"] * 2,
        "model_checkpoint": "evo2_7b",
        "local_path": None,
        "top_k": 2,
        "top_p": 0.8,
        "temperature": 0.7,
        "max_new_tokens": 4,
        "cached_generation": True,
        "force_prompt_threshold": 2048,
        "max_seqlen": None,
        "skip_special_tokens": False,
        "stop_at_eos": False,
        "old_kv_cache": None,
        "return_kv_cache": False,
        "return_logits": False,
        "batch_size": 2,
        "device": "cuda",
        "verbose": False,
        "seed": seed,
    }
    assert result.metadata["seed"] == seed


@pytest.mark.parametrize(("batch_size", "threshold"), [(1, 4096), (2, 2048), (3, 1024), (8, 512)])
def test_evo2_icl_bounds_parallel_prefill_without_truncating_context(mock_sampler, batch_size, threshold):
    """Batch-aware prefill stays a power of two while forwarding the complete prompt."""
    calls, sampled = mock_sampler
    sampled.append("ACGT")
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(
        inputs,
        Evo2ICLConfig(num_sequences=1, sequence_length=4, batch_size=batch_size, device="cuda"),
    )

    payload = calls[0][1]
    assert payload["force_prompt_threshold"] == threshold
    assert threshold * batch_size <= 4096
    assert payload["prompts"] == [("N" + "N".join(inputs.examples) + "N")]
    assert len(payload["prompts"][0]) == 32_768
    assert result.metadata["prefill_threshold"] == threshold


def test_evo2_icl_filters_invalid_generations_with_yield_warning(mock_sampler):
    """Only requested-length DNA continuations survive, including sampled N bases."""
    _, sampled = mock_sampler
    sampled.extend(["AAAA", "ACGN", "AC", "ACGTAC", "TTTT", "acgt"])

    result = run_evo2_icl(
        Evo2ICLInput(examples=_EXAMPLES),
        Evo2ICLConfig(num_sequences=6, sequence_length=4, device="cuda"),
    )

    assert result.sequences == ["AAAA", "ACGN", "TTTT"]
    assert result.metadata["num_requested"] == 6
    assert result.metadata["num_generated"] == 3
    assert result.metadata["num_omitted"] == 3
    assert len(result.warnings) == 1
    assert "Returned 3 of 6 requested designs" in result.warnings[0]
    assert "3 generations were empty, invalid, or did not meet the length requirement" in result.warnings[0]


def test_evo2_icl_rejects_all_invalid_generations(mock_sampler):
    """A run without any usable continuation raises an actionable error."""
    _, sampled = mock_sampler
    sampled.extend(["XXXX", ""])

    with pytest.raises(ValueError, match="No usable A/C/G/T/N designs were generated"):
        run_evo2_icl(
            Evo2ICLInput(examples=_EXAMPLES),
            Evo2ICLConfig(num_sequences=2, sequence_length=4, device="cuda"),
        )


@pytest.mark.parametrize("sequences", [[], ["AAAA"], ["AAAA", "CCCC", "GGGG"]])
def test_evo2_icl_rejects_wrong_sampler_result_count(mock_sampler, sequences):
    """A malformed sampler response cannot silently change the requested workload."""
    _, sampled = mock_sampler
    sampled.extend(sequences)

    with pytest.raises(ValueError, match=f"Evo 2 returned {len(sequences)} generations for 2 requested designs"):
        run_evo2_icl(
            Evo2ICLInput(examples=_EXAMPLES),
            Evo2ICLConfig(num_sequences=2, sequence_length=4, device="cuda"),
        )


def test_export_fasta_contains_only_generated_designs(mock_sampler, tmp_path):
    """Exported FASTA excludes examples, separators, and invalid continuations."""
    from Bio import SeqIO

    _, sampled = mock_sampler
    sampled.extend(["AAAA", "XCCC", "TTTT"])
    result = run_evo2_icl(
        Evo2ICLInput(examples=_EXAMPLES),
        Evo2ICLConfig(num_sequences=3, sequence_length=4, device="cuda"),
    )

    result.export("icl_designs", export_path=tmp_path, file_format="fasta")

    records = list(SeqIO.parse(tmp_path / "icl_designs.fasta", "fasta"))
    assert [str(record.seq) for record in records] == ["AAAA", "TTTT"]
    assert len({record.id for record in records}) == 2


def test_evo2_icl_generates_one_thousand_designs(mock_sampler):
    """Sixteen 200-bp examples can condition the full allowance with the default batch."""
    calls, sampled = mock_sampler
    sampled.extend(("ACGT"[i % 4] * 200) for i in range(1000))

    result = run_evo2_icl(
        Evo2ICLInput(examples=_TWO_HUNDRED_BP_EXAMPLES),
        Evo2ICLConfig(num_sequences=1000, sequence_length=200, fixed_examples=True, device="cuda"),
    )

    assert len(calls) == 1
    assert len(calls[0][1]["prompts"]) == 1000
    assert set(calls[0][1]["prompts"]) == {"N" + "N".join(_TWO_HUNDRED_BP_EXAMPLES) + "N"}
    assert calls[0][1]["batch_size"] == 1
    assert calls[0][1]["force_prompt_threshold"] == 4096
    assert result.sequences == sampled
    assert len(result.results) == 1000


def test_evo2_icl_full_context_is_preserved_with_long_design(mock_sampler):
    """A full prompt remains intact when conditioning a long design within the work limit."""
    calls, sampled = mock_sampler
    sampled.append("A" * 4096)
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=1, sequence_length=4096, device="cuda"))

    assert len(calls[0][1]["prompts"][0]) == 32_768
    assert calls[0][1]["max_new_tokens"] == 4096
    assert len(result.sequences[0]) == 4096


def test_evo2_icl_large_local_run_reaches_sampler_without_count_or_work_quota(mock_sampler):
    """Local dispatch can exceed both the old count and sequential-work limits."""
    calls, sampled = mock_sampler
    sampled.extend(["A"] * 1001)
    config = Evo2ICLConfig(num_sequences=1001, sequence_length=1, device="cuda")
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(inputs, config)

    assert len(result.results) == 1001
    assert len(calls) == 1
    assert len(calls[0][1]["prompts"]) == 1001
    assert all(len(prompt) == 32_768 for prompt in calls[0][1]["prompts"])


@pytest.mark.parametrize(
    ("spacer", "both", "prompt"),
    [("N", False, "NACGTNTGCAN"), ("NN", True, "NNACGTNNNNTGCANNNN"), ("", False, "ACGTTGCA")],
)
def test_evo2_icl_matches_reference_spacer_composition(spacer, both, prompt):
    """Match Evo2_ICL.py's leading/trailing spacers, including doubled wrap boundaries."""
    config = Evo2ICLConfig(n_shots=2, fixed_examples=True, spacer=spacer, spacer_both_sides=both)
    assert config.compose_prompts(Evo2ICLInput(examples=["ACGT", "TGCA", "AAAA"])) == ([prompt], [[1, 2]])


def test_evo2_icl_reference_example_draws_are_reproducible_and_independent_of_generation():
    """Reference RandomState(42) draws advance across prompts without replacement."""
    inputs = Evo2ICLInput(examples=["A", "C", "G", "T", "AA"])
    config = Evo2ICLConfig(n_shots=3, n_prompts=2, example_seed=42)
    prompts, selected = config.compose_prompts(inputs)
    assert selected == [[2, 5, 3], [4, 2, 3]]
    assert prompts == ["NCNAANGN", "NTNCNGN"]
    assert config.compose_prompts(inputs) == (prompts, selected)
    assert config.model_copy(update={"seed": 123}).compose_prompts(inputs) == (prompts, selected)
    assert config.model_copy(update={"example_seed": 43}).compose_prompts(inputs) != (prompts, selected)


def test_evo2_icl_large_example_pool_is_subsampled(mock_sampler):
    """A HepG2-sized pool fits by selecting examples, without truncating any record."""
    calls, sampled = mock_sampler
    sampled.extend(["ACGT"] * 2)
    result = run_evo2_icl(
        Evo2ICLInput(examples=["ACGT" * 50] * 463),
        Evo2ICLConfig(n_shots=16, num_sequences=2, sequence_length=4, device="cuda"),
    )
    assert all(len(prompt) == 3217 for prompt in calls[0][1]["prompts"])
    assert result.metadata["num_examples"] == 463
    assert len(set(result.metadata["prompt_example_indices"][0])) == 16


def test_evo2_icl_multiple_prompts_use_one_dispatch_and_preserve_output_provenance(mock_sampler):
    """Prompt/sample ordering survives omission of invalid generations, with one RNG execution."""
    calls, sampled = mock_sampler
    sampled.extend(["AAAA", "XXXX", "CCCC", "TTTT"])
    config = Evo2ICLConfig(n_shots=1, n_prompts=2, num_sequences=2, sequence_length=4, seed=37, device="cuda")
    inputs = Evo2ICLInput(examples=["ACGT", "TGCA", "CCCC"])
    prompts, selections = config.compose_prompts(inputs)
    result = run_evo2_icl(inputs, config)
    assert len(calls) == 1
    assert calls[0][1]["prompts"] == [prompts[0], prompts[0], prompts[1], prompts[1]]
    assert calls[0][1]["seed"] == 37
    assert result.sequences == ["AAAA", "CCCC", "TTTT"]
    assert result.metadata["num_requested"] == 4
    assert result.metadata["prompt_example_indices"] == selections
    assert result.metadata["design_prompt_indices"] == [1, 2, 2]
    assert "Returned 3 of 4" in result.warnings[0]


def test_evo2_icl_fixed_examples_clamp_shots_and_repeat_order_across_prompts():
    """Oversized shot counts use every available example, matching the reference script."""
    config = Evo2ICLConfig(n_shots=16, n_prompts=2, fixed_examples=True)
    assert config.compose_prompts(Evo2ICLInput(examples=["A", "C"])) == (["NANCN"] * 2, [[1, 2]] * 2)


def test_evo2_icl_minimum_length_checks_entire_pool_before_dispatch(mock_sampler):
    """An unselected short record still fails validation, as in the reference FASTA loader."""
    calls, _ = mock_sampler
    with pytest.raises(ValueError, match="Example 2 is shorter than min_example_len=4"):
        run_evo2_icl(
            Evo2ICLInput(examples=["ACGT", "A"]),
            Evo2ICLConfig(n_shots=1, fixed_examples=True, min_example_len=4, device="cuda"),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"force_prompt_threshold": 1000}, "power of two"),
        ({"spacer": "<SEP>"}, "separator must contain only"),
    ],
)
def test_evo2_icl_rejects_invalid_combined_controls(kwargs, message):
    """Combined limits and delimiter semantics are checked before any model execution."""
    with pytest.raises(ValidationError, match=message):
        Evo2ICLConfig(**kwargs)


def test_evo2_icl_explicit_prefill_reaches_fixed_length_sampler(mock_sampler):
    """Explicit prefill is forwarded while early stopping stays disabled."""
    calls, sampled = mock_sampler
    sampled.append("ACGT")
    config = Evo2ICLConfig(num_sequences=1, sequence_length=4, force_prompt_threshold=1024, device="cuda")
    run_evo2_icl(Evo2ICLInput(examples=["ACGT"]), config)
    assert calls[0][1]["force_prompt_threshold"] == 1024
    assert calls[0][1]["stop_at_eos"] is False


def test_evo2_icl_requested_length_ignores_separator(mock_sampler):
    """A generated separator is DNA content when a fixed length was requested."""
    calls, sampled = mock_sampler
    sampled.append("ACNNTG")
    result = run_evo2_icl(
        Evo2ICLInput(examples=["ACGT"]),
        Evo2ICLConfig(num_sequences=1, sequence_length=6, spacer="NN", device="cuda"),
    )
    assert result.sequences == ["ACNNTG"]
    assert calls[0][1]["max_new_tokens"] == 6
    assert "stop_sequence" not in calls[0][1]


def test_evo2_icl_requires_full_requested_length(mock_sampler):
    """A truncated result is never accepted as a requested-length design."""
    _, sampled = mock_sampler
    sampled.extend(["AC", "ACNT"])
    result = run_evo2_icl(
        Evo2ICLInput(examples=["ACGT"]),
        Evo2ICLConfig(num_sequences=2, sequence_length=4, device="cuda"),
    )
    assert result.sequences == ["ACNT"]
    assert result.metadata["num_omitted"] == 1


def test_evo2_icl_length_cannot_be_cleared():
    """The generation loop always receives a positive finite token limit."""
    with pytest.raises(ValidationError, match="valid integer"):
        Evo2ICLConfig(sequence_length=None)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Evo2ICLConfig(sequence_length=0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark("evo2-icl")
@pytest.mark.uses_gpu
def test_evo2_icl_benchmark(request):
    """Benchmark two 200-bp pretrained-7B designs, checking cold and warm worker outputs."""
    inputs = Evo2ICLInput(examples=_TWO_HUNDRED_BP_EXAMPLES)
    config = Evo2ICLConfig(num_sequences=2, sequence_length=200, verbose=False)
    result = benchmark_twice(request, "evo2", lambda: run_evo2_icl(inputs, config))

    assert result.tool_id == "evo2-icl"
    assert result.metadata["model_checkpoint"] == "evo2_7b"
    assert 1 <= len(result.sequences) <= 2
    assert all(len(sequence) == 200 and set(sequence) <= set("ACGTN") for sequence in result.sequences)
    assert result.metadata["num_requested"] == 2
    assert result.metadata["num_generated"] == len(result.sequences)


def test_standard_code_matches_ncbi_table_one():
    """Every codon agrees with the independently maintained standard-code table."""
    from Bio.Data.CodonTable import unambiguous_dna_by_id

    reference = unambiguous_dna_by_id[1]
    expected = {**reference.forward_table, **dict.fromkeys(reference.stop_codons, "*")}
    assert expected == evo2_icl._CODON_TO_AA


def test_protein_reverse_translation_roundtrip_and_seed():
    """Uniform codon draws preserve residues, reuse the seed, and advance across records."""
    from Bio.Seq import Seq

    protein = "ACDEFGHIKLMNPQRSTVWY" * 10
    inputs = Evo2ICLInput(examples=[protein.lower(), protein], sequence_type="protein")
    dna = inputs.dna_examples(seed=17)
    assert [str(Seq(sequence).translate()) for sequence in dna] == [protein, protein]
    assert inputs.dna_examples(seed=17) == dna
    assert inputs.dna_examples(seed=18) != dna
    assert dna[0] != dna[1]
    assert inputs.examples == [protein, protein]
    assert inputs.dna_examples(seed=None) != inputs.dna_examples(seed=None)


def test_reverse_translation_uses_all_synonymous_codons_uniformly():
    """A fixed large draw exercises every leucine codon without a preferred-codon bias."""
    from collections import Counter

    dna = Evo2ICLInput(examples=["L" * 6000], sequence_type="protein").dna_examples(seed=17)[0]
    counts = Counter(dna[i : i + 3] for i in range(0, len(dna), 3))
    assert set(counts) == {"TTA", "TTG", "CTT", "CTC", "CTA", "CTG"}
    assert all(800 < count < 1200 for count in counts.values())


@pytest.mark.parametrize("sequence", ["MAX", "MA*", "MAB", "MAZ", "MAU", "MAO", "MA-", ""])
def test_protein_input_rejects_ambiguous_or_stop_residues(sequence):
    """Reverse translation requires an unambiguous amino acid at each input position."""
    with pytest.raises(ValidationError, match="20 standard amino acids"):
        Evo2ICLInput(examples=[sequence], sequence_type="protein")


def test_protein_prompts_keep_selection_separate_from_codon_seed():
    """Changing codon seed changes DNA without changing example selection or record ordering."""
    inputs = Evo2ICLInput(examples=["LN" * 20, "AL" * 20, "RG" * 20], sequence_type="protein")
    first = Evo2ICLConfig(n_shots=2, n_prompts=2, codon_seed=3, example_seed=19)
    second = Evo2ICLConfig(n_shots=2, n_prompts=2, codon_seed=4, example_seed=19)
    prompts, selected = first.compose_prompts(inputs)
    other_prompts, other_selected = second.compose_prompts(inputs)
    assert first.model_copy(update={"seed": 73}).compose_prompts(inputs) == (prompts, selected)
    assert selected == other_selected
    assert prompts != other_prompts
    dna = inputs.dna_examples(seed=3)
    assert prompts == ["N" + "N".join(dna[i - 1] for i in indices) + "N" for indices in selected]


def test_protein_generation_returns_translated_fasta_and_matching_dna(mock_sampler, tmp_path):
    """The sampler generates three bases per residue; stops/N never truncate the translation."""
    from Bio.Seq import Seq

    calls, sampled = mock_sampler
    sampled.extend(["ATGTAANNNTGG", "GCTTGTTTTGAA", "ATG"])
    inputs = Evo2ICLInput(examples=[" m\nlnw "], sequence_type="protein")
    config = Evo2ICLConfig(num_sequences=3, sequence_length=4, seed=17, device="cuda")
    result = run_evo2_icl(inputs, config)
    assert result.success
    payload = calls[0][1]
    assert payload["max_new_tokens"] == 12
    assert payload["stop_at_eos"] is False
    assert payload["seed"] == 17
    assert str(Seq(payload["prompts"][0][1:-1]).translate()) == "MLNW"
    assert result.sequences == ["M*XW", "ACFE"]
    assert result.dna_sequences == sampled[:2]
    assert result.sequence_type == "protein"
    assert result.metadata["design_prompt_indices"] == [1, 1]
    assert result.metadata["num_omitted"] == 1
    assert result.metadata["generated_bases_per_design"] == 12
    assert any("stop codons" in warning for warning in result.warnings)
    assert any("ambiguous codons" in warning for warning in result.warnings)
    result.export("proteins", export_path=tmp_path, file_format="fasta")
    assert "M*XW" in (tmp_path / "proteins.fasta").read_text()


def test_protein_codon_and_generation_seeds_are_independent(mock_sampler):
    """Separate seeds control prompt DNA and decoder sampling, including cache keys."""
    calls, sampled = mock_sampler
    sampled.extend(["ATG" * 4])
    inputs = Evo2ICLInput(sequence_type="protein", examples=["ACDEFGHIKLMNPQRSTVWY" * 2])
    config = Evo2ICLConfig(num_sequences=1, sequence_length=4, seed=17, codon_seed=23, device="cuda")
    for changed in [config, config.model_copy(update={"seed": 18}), config.model_copy(update={"codon_seed": 24})]:
        result = run_evo2_icl(inputs, changed)
        assert result.metadata["seed"] == changed.seed
        assert result.metadata["codon_seed"] == changed.codon_seed
    assert len(calls) == 3
    first, generation_changed, codons_changed = [call[1] for call in calls]
    assert first["prompts"] == generation_changed["prompts"]
    assert first["prompts"] != codons_changed["prompts"]
    assert [payload["seed"] for payload in [first, generation_changed, codons_changed]] == [17, 18, 17]
