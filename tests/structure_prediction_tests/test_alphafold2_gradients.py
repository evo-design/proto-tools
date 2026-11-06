"""tests/structure_prediction_tests/test_alphafold2_gradients.py.

Tests for the three AlphaFold2 design-gradient tools and the machinery they share.
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from proto_tools.entities.structures import BFactorType, Structure, is_valid_structure
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BetaStrandTerm,
    AlphaFold2BinderGradientCandidate,
    AlphaFold2BinderGradientConfig,
    AlphaFold2BinderGradientInput,
    AlphaFold2BinderRedesignGradientCandidate,
    AlphaFold2BinderRedesignGradientConfig,
    AlphaFold2BinderRedesignGradientInput,
    AlphaFold2FixBBGradientCandidate,
    AlphaFold2FixBBGradientConfig,
    AlphaFold2FixBBGradientInput,
    AlphaFold2HallucinationGradientConfig,
    AlphaFold2HallucinationGradientInput,
    AlphaFold2HallucinationGradientOutput,
    AlphaFold2HallucinationGradientResult,
    AlphaFold2HallucinationTermGradient,
    AlphaFold2InterfaceContactTerm,
    AlphaFold2PlddtTerm,
    AlphaFold2RmsdTerm,
    gradient_helpers,
    run_alphafold2_binder_gradient,
    run_alphafold2_binder_redesign_gradient,
    run_alphafold2_fixbb_gradient,
    run_alphafold2_hallucination_gradient,
)
from proto_tools.tools.structure_prediction.alphafold2.gradient_helpers import build_terms
from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import PROTEIN_AMINO_ACIDS, ProteinSequenceLogits, one_hot_protein_logits
from proto_tools.utils.gradient.models import TermGradient
from tests.conftest import benchmark_twice

_EXAMPLE_PDB = (
    Path(__file__).resolve().parents[2]
    / "proto_tools/tools/structure_prediction/alphafold2/examples/example_output/alphafold2_structure/structure_0.pdb"
).read_text()

_FIXTURE_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"

_CONFIGS = {
    "hallucination": AlphaFold2HallucinationGradientConfig,
    "fixbb": AlphaFold2FixBBGradientConfig,
    "binder": AlphaFold2BinderGradientConfig,
    "binder-redesign": AlphaFold2BinderRedesignGradientConfig,
}
_KEYS = {
    "hallucination": "alphafold2-hallucination-gradient",
    "fixbb": "alphafold2-fixbb-gradient",
    "binder": "alphafold2-binder-gradient",
    "binder-redesign": "alphafold2-binder-redesign-gradient",
}


@pytest.fixture
def target() -> Structure:
    """The shared PD-L1 fixture structure."""
    return Structure(structure=_FIXTURE_PDB.read_text())


@pytest.mark.parametrize("protocol", sorted(_CONFIGS))
def test_duplicate_terms_rejected(protocol: str) -> None:
    """A quantity yields one gradient, so naming it twice is a mistake rather than a doubling."""
    with pytest.raises(ValueError, match="more than once"):
        _CONFIGS[protocol](gradient_terms=["plddt", "plddt"])


def test_a_term_may_repeat_with_different_selectors() -> None:
    """Germinal scores pLDDT over the whole binder and over its CDRs; each is its own gradient."""
    whole, cdrs = AlphaFold2PlddtTerm(), AlphaFold2PlddtTerm(positions=[2, 3])
    config = AlphaFold2BinderRedesignGradientConfig(gradient_terms=[whole, cdrs])
    assert config.gradient_terms == [whole, cdrs]


@pytest.mark.parametrize("protocol", sorted(_CONFIGS))
def test_terms_outside_the_protocol_catalog_are_rejected(protocol: str) -> None:
    """Each protocol offers only the quantities it actually computes."""
    offered = set(_CONFIGS[protocol].supported_terms())
    assert ("rmsd" in offered) is (protocol == "fixbb")
    assert ("i_pae" in offered) is protocol.startswith("binder")
    absent = "rmsd" if protocol.startswith("binder") else "i_pae"
    with pytest.raises(ValueError):
        _CONFIGS[protocol](gradient_terms=[absent])


@pytest.mark.parametrize("protocol", sorted(_CONFIGS))
def test_probability_mode_rejects_knobs_it_would_ignore(protocol: str) -> None:
    """temperature=None bypasses the softmax, so soft and hard would silently do nothing."""
    config = _CONFIGS[protocol](soft=0.5, hard=1.0)
    with pytest.raises(ValueError, match="would have no effect"):
        gradient_helpers._reject_unused_softmax_knobs(config, None, _KEYS[protocol])
    # The same knobs are meaningful once a temperature is supplied.
    gradient_helpers._reject_unused_softmax_knobs(config, 1.0, _KEYS[protocol])


def test_from_probabilities_round_trips_with_context_fields(target: Structure) -> None:
    """Context-bearing candidates must survive the documented probability entry point."""
    probabilities = (np.ones((6, 20)) / 20).tolist()

    plain = AlphaFold2HallucinationGradientInput.from_probabilities(probabilities)
    fixbb = AlphaFold2FixBBGradientInput.from_probabilities(probabilities, reference_pdb=target, reference_chain="B")
    binder = AlphaFold2BinderGradientInput.from_probabilities(probabilities, target_pdb=target, target_chain="A")

    assert fixbb.inputs[0].reference_chain == "B"
    assert binder.inputs[0].target_chain == "A"
    for built in (plain, fixbb, binder):
        assert len(built.inputs) == 1


@pytest.mark.parametrize(
    ("cls", "field", "context"),
    [
        (AlphaFold2FixBBGradientInput, "reference_pdb", {}),
        (AlphaFold2BinderGradientInput, "target_pdb", {}),
    ],
)
def test_from_probabilities_requires_its_structure(cls: Any, field: str, context: dict[str, Any]) -> None:
    """The structure belongs to the candidate, so omitting it is an error rather than a default."""
    with pytest.raises(ValueError, match="requires the"):
        cls.from_probabilities((np.ones((4, 20)) / 20).tolist(), **context)


def test_target_chain_is_stored_as_colabdesign_parses_it(target: Structure) -> None:
    """ColabDesign splits on commas without stripping, so the stored form must carry no spaces."""
    candidate = AlphaFold2BinderGradientCandidate(logits=[[0.0] * 20] * 6, target_pdb=target, target_chain="A, B")
    assert candidate.target_chain == "A,B"


def test_absent_target_chain_is_rejected(target: Structure) -> None:
    """A chain that is not in the structure would otherwise be dropped silently by ColabDesign."""
    with pytest.raises(ValueError, match="absent from target_pdb"):
        AlphaFold2BinderGradientCandidate(logits=[[0.0] * 20] * 6, target_pdb=target, target_chain="Z")


def test_build_terms_shares_one_basis_and_each_term_adds_what_only_it_knows() -> None:
    """Metadata is computed once per candidate; the term supplies its own description and units."""
    built = build_terms(
        [
            {"id": "rmsd", "value": 1.5, "gradient": [[0.1] * 20] * 3},
            {"id": "plddt", "value": 0.8, "gradient": [[0.2] * 20] * 3},
        ],
        [AlphaFold2RmsdTerm(), AlphaFold2PlddtTerm()],
        [[0.0] * 20] * 3,
        temperature=1.0,
        soft=1.0,
        hard=0.0,
    )
    assert built[0]["metadata"].input_fingerprint == built[1]["metadata"].input_fingerprint, "one basis"

    # The quantity's own facts are overlaid when the gradient is built, not by the caller.
    rmsd, plddt = (TermGradient(**kwargs) for kwargs in built)
    assert rmsd.metadata.units == "angstroms"
    assert plddt.metadata.units is None
    assert rmsd.metadata.description == AlphaFold2RmsdTerm.DESCRIPTION
    assert plddt.metadata.description == AlphaFold2PlddtTerm.DESCRIPTION
    assert rmsd.metadata.description != plddt.metadata.description


def test_a_term_with_nothing_to_measure_is_named_in_the_error() -> None:
    """An empty aux array reaches the wrapper as a null value; say which quantity, not float(None)."""
    with pytest.raises(ValueError, match="plddt"):
        build_terms(
            [{"id": "plddt", "value": None, "gradient": None}],
            [AlphaFold2PlddtTerm()],
            [[0.0] * 20] * 3,
            temperature=1.0,
            soft=1.0,
            hard=0.0,
        )


def test_an_unrequested_term_is_named_in_the_error() -> None:
    """A worker reporting a quantity nobody asked for should say so, not raise KeyError."""
    with pytest.raises(ValueError, match="i_con"):
        build_terms(
            [{"id": "i_con", "value": 0.5, "gradient": None}],
            [AlphaFold2PlddtTerm()],
            [[0.0] * 20] * 3,
            temperature=1.0,
            soft=1.0,
            hard=0.0,
        )


def test_helix_requires_an_explicit_goal() -> None:
    """More helicity is not universally better, so composing helix must not assume a direction."""
    term = gradient_helpers.AlphaFold2HelixTerm()
    gradient = TermGradient(
        term=term,
        term_value=0.5,
        gradient=[[1.0] + [0.0] * 19] * 3,
        metadata=build_terms(
            [{"id": "helix", "value": 0.5, "gradient": [[1.0] + [0.0] * 19] * 3}],
            [gradient_helpers.AlphaFold2HelixTerm()],
            [[0.0] * 20] * 3,
            temperature=1.0,
            soft=1.0,
            hard=0.0,
        )[0]["metadata"],
    )
    with pytest.raises(ValueError, match="needs an explicit goal"):
        gradient.apply_goal()


@pytest.mark.parametrize("protocol", sorted(_KEYS))
def test_tools_declare_themselves_stochastic(protocol: str) -> None:
    """config.seed reaches the model, so the framework must re-seed each chunk of a batch."""
    assert ToolRegistry.get(_KEYS[protocol]).stochastic is True


def test_registry_descriptions_are_distinct() -> None:
    """Agents select between these three on the description alone."""
    descriptions = {key: ToolRegistry.get(key).description for key in _KEYS.values()}
    assert len(set(descriptions.values())) == len(descriptions)


@pytest.mark.parametrize("protocol", sorted(_CONFIGS))
def test_probability_mode_rejects_excluded_amino_acids(protocol: str) -> None:
    """The exclusion is a bias on the softmax branch, which probability mode never uses."""
    config = _CONFIGS[protocol](excluded_amino_acids=["C"])
    with pytest.raises(ValueError, match="excluded_amino_acids"):
        gradient_helpers._reject_unused_softmax_knobs(config, None, _KEYS[protocol])
    gradient_helpers._reject_unused_softmax_knobs(config, 1.0, _KEYS[protocol])


def test_binder_dispatch_forwards_its_target_controls(monkeypatch: pytest.MonkeyPatch, target: Structure) -> None:
    """The binder protocol is only comparable across runs if the template masking reaches prep."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(payload)
        return {
            "results": [
                {
                    "terms": [{"id": "plddt", "value": 0.5, "gradient": [[0.0] * 20] * 6}],
                    "metrics": {"avg_plddt": 0.5},
                    "pdb": _FIXTURE_PDB.read_text(),
                }
            ]
        }

    monkeypatch.setattr(gradient_helpers.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    run_alphafold2_binder_gradient(
        AlphaFold2BinderGradientInput(
            inputs=[AlphaFold2BinderGradientCandidate(logits=[[0.0] * 20] * 6, target_pdb=target, target_chain="A")]
        ),
        AlphaFold2BinderGradientConfig(gradient_terms=["plddt"]),
    )

    assert captured["protocol"] == "binder"
    assert captured["use_multimer"] is True
    prep = captured["items"][0]["prep_kwargs"]
    assert prep["rm_target_seq"] is True
    assert prep["rm_template_ic"] is True
    assert prep["rm_target_sc"] is False
    assert prep["binder_len"] == 6


def test_fixbb_dispatch_sends_the_reference_chain(monkeypatch: pytest.MonkeyPatch, target: Structure) -> None:
    """Fixbb takes its length from the reference, so the chain selection must reach prep_inputs."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(payload)
        return {
            "results": [
                {
                    "terms": [{"id": "rmsd", "value": 2.0, "gradient": [[0.0] * 20] * 4}],
                    "metrics": {"avg_plddt": 0.5},
                    "pdb": _FIXTURE_PDB.read_text(),
                }
            ]
        }

    monkeypatch.setattr(gradient_helpers.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    output = run_alphafold2_fixbb_gradient(
        AlphaFold2FixBBGradientInput(
            inputs=[
                AlphaFold2FixBBGradientCandidate(logits=[[0.0] * 20] * 4, reference_pdb=target, reference_chain="B")
            ]
        ),
        AlphaFold2FixBBGradientConfig(gradient_terms=["rmsd"]),
    )

    assert captured["protocol"] == "fixbb"
    assert captured["items"][0]["prep_kwargs"]["chain"] == "B"
    assert "use_multimer" not in captured
    (result,) = output.results
    assert result.terms[0].term.id == "rmsd"
    assert result.terms[0].metadata.units == "angstroms"


def test_binder_selectors_reach_the_worker_and_come_back_with_the_term(
    monkeypatch: pytest.MonkeyPatch, target: Structure
) -> None:
    """A CDR-restricted term must reach the worker and be reported as the term that was asked for."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(payload)
        terms = [{"id": term_id, "value": 0.5, "gradient": [[0.0] * 20] * 6} for term_id in payload["gradient_terms"]]
        return {"results": [{"terms": terms, "metrics": {"avg_plddt": 0.5}, "pdb": _FIXTURE_PDB.read_text()}]}

    monkeypatch.setattr(gradient_helpers.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    i_con = AlphaFold2InterfaceContactTerm(positions=[2, 3], num_contacts=2, cutoff=20.0)
    strand = AlphaFold2BetaStrandTerm(positions=[1, 2, 3, 4, 5], form="out_of_range")
    output = run_alphafold2_binder_gradient(
        AlphaFold2BinderGradientInput(
            inputs=[AlphaFold2BinderGradientCandidate(logits=[[0.0] * 20] * 6, target_pdb=target, target_chain="A")]
        ),
        AlphaFold2BinderGradientConfig(gradient_terms=[i_con, strand]),
    )

    assert captured["term_options"][0] == {"positions": [2, 3], "num_contacts": 2, "cutoff": 20.0}
    assert captured["term_options"][1]["form"] == "out_of_range"
    assert [term.term for term in output.results[0].terms] == [i_con, strand]


def test_redesign_prepares_the_template_binder_chain(monkeypatch: pytest.MonkeyPatch, target: Structure) -> None:
    """Redesign must hand ColabDesign the binder chain with its backbone kept and sequence masked."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(payload)
        terms = [{"id": "plddt", "value": 0.8, "gradient": [[0.0] * 20] * 4}]
        return {"results": [{"terms": terms, "metrics": {"avg_plddt": 0.8}, "pdb": _FIXTURE_PDB.read_text()}]}

    monkeypatch.setattr(gradient_helpers.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    candidate = AlphaFold2BinderRedesignGradientCandidate(
        logits=[[0.0] * 20] * 4, template_pdb=target, target_chain="A", binder_chain="B"
    )
    run_alphafold2_binder_redesign_gradient(
        AlphaFold2BinderRedesignGradientInput(inputs=[candidate]), AlphaFold2BinderRedesignGradientConfig()
    )

    prep = captured["items"][0]["prep_kwargs"]
    assert captured["protocol"] == "binder"
    assert prep["binder_chain"] == "B" and "binder_len" not in prep
    assert (prep["rm_binder"], prep["rm_binder_seq"], prep["rm_binder_sc"], prep["rm_template_ic"]) == (
        False,
        True,
        True,
        True,
    )


def test_redesign_rejects_a_binder_chain_that_is_also_the_target(target: Structure) -> None:
    """Redesigning a target chain against itself is a misnamed chain, not a design."""
    with pytest.raises(ValueError, match="also named as a target chain"):
        AlphaFold2BinderRedesignGradientCandidate(
            logits=[[0.0] * 20] * 4, template_pdb=target, target_chain="A,B", binder_chain="B"
        )


@pytest.mark.uses_gpu
def test_fixbb_rejects_logits_that_do_not_cover_the_reference_chain(target: Structure) -> None:
    """Short logits were injected into the trailing rows, folding a sequence nobody supplied."""
    chain_length = len(target.get_chain_sequence("B"))
    candidate = AlphaFold2FixBBGradientCandidate(logits=[[0.0] * 20] * 5, reference_pdb=target, reference_chain="B")
    assert chain_length != 5
    # The worker owns this check, so the failure arrives as the subprocess error.
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_alphafold2_fixbb_gradient(
            AlphaFold2FixBBGradientInput(inputs=[candidate]),
            AlphaFold2FixBBGradientConfig(gradient_terms=["rmsd"]),
        )
    assert f"supplies 5 residues but the fixbb protocol designs {chain_length}" in raised.value.stderr


# -- Benchmarks ---------------------------------------------------------------

#: Repeated passes: these run in an optimization loop, so warm per-step cost is what matters.
_BENCHMARK_STEPS = 5


def _run_steps(runner: Any) -> Any:
    """Drive ``runner`` through the benchmark's repeated passes, returning the last result."""
    last = None
    for _ in range(_BENCHMARK_STEPS):
        last = runner()
    return last


def _assert_gradient_result(output: Any, *, expected_terms: list[str], length: int) -> None:
    """Every benchmarked protocol owes one finite gradient per requested term, plus a structure."""
    assert output.success
    (result,) = output.results
    assert [gradient.term.id for gradient in result.terms] == expected_terms
    for gradient in result.terms:
        assert math.isfinite(gradient.term_value)
        assert gradient.gradient is not None
        assert len(gradient.gradient) == length
        assert all(len(row) == 20 for row in gradient.gradient)
        assert all(math.isfinite(value) for row in gradient.gradient for value in row)
    assert is_valid_structure(result.structure.structure)


@pytest.mark.parametrize("name", ["step", "step_v1.5"])
def test_gradient_export_writes_one_pdb_sidecar_per_result(tmp_path: Path, name: str) -> None:
    """Each result gets its own indexed sidecar, so a batch cannot overwrite one structure."""
    structure = Structure(structure=_EXAMPLE_PDB, b_factor_type=BFactorType.PLDDT)
    AlphaFold2HallucinationGradientOutput(
        results=[
            AlphaFold2HallucinationGradientResult(
                terms=[
                    AlphaFold2HallucinationTermGradient(
                        **build_terms(
                            [{"id": "plddt", "value": value, "gradient": [[0.1] * 20] * 2}],
                            [AlphaFold2PlddtTerm()],
                            [[0.0] * 20] * 2,
                            temperature=1.0,
                            soft=1.0,
                            hard=0.0,
                        )[0]
                    )
                ],
                metrics={"avg_plddt": value},
                structure=structure,
            )
            for value in (0.8, 0.6)
        ],
        vocab=list(PROTEIN_AMINO_ACIDS),
    ).export(name, export_path=tmp_path)

    payload = json.loads((tmp_path / f"{name}.json").read_text())
    assert payload["structure_pdbs"] == [f"{name}_0000.pdb", f"{name}_0001.pdb"]
    for sidecar in payload["structure_pdbs"]:
        assert is_valid_structure((tmp_path / sidecar).read_text())


@pytest.mark.benchmark("alphafold2-hallucination-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphafold2_hallucination_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark the hallucination protocol: 5 passes over a 118 aa free sequence (cold + warm).

    Two terms rather than one, because the per-term VJP takes a separate pullback for each and a
    single-term run would not show that cost. The sequence is the PD-L1 nanobody so the three
    protocol benchmarks are comparable at equal length.
    """
    sequence = Structure.from_file(_FIXTURE_PDB).get_chain_sequence("B")
    inputs = AlphaFold2HallucinationGradientInput(
        inputs=[ProteinSequenceLogits(one_hot_protein_logits(sequence, sharpness=2.0))],
        temperature=1.0,
    )
    config = AlphaFold2HallucinationGradientConfig(gradient_terms=["plddt", "con"], num_recycles=1, seed=0)

    result = benchmark_twice(
        request, "alphafold2", lambda: _run_steps(lambda: run_alphafold2_hallucination_gradient(inputs, config))
    )

    _assert_gradient_result(result, expected_terms=["plddt", "con"], length=len(sequence))


@pytest.mark.benchmark("alphafold2-fixbb-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphafold2_fixbb_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark the fixbb protocol: 5 redesign passes against the PD-L1 nanobody backbone (cold + warm).

    ``rmsd`` is included alongside ``plddt`` because it is the term unique to this protocol and the
    only one whose gradient flows through the structural alignment.
    """
    reference = Structure.from_file(_FIXTURE_PDB)
    length = len(reference.get_chain_sequence("B"))
    inputs = AlphaFold2FixBBGradientInput(
        inputs=[
            AlphaFold2FixBBGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits("A" * length, sharpness=2.0)),
                reference_pdb=reference,
                reference_chain="B",
            )
        ],
        temperature=1.0,
    )
    config = AlphaFold2FixBBGradientConfig(gradient_terms=["plddt", "rmsd"], num_recycles=1, seed=0)

    result = benchmark_twice(
        request, "alphafold2", lambda: _run_steps(lambda: run_alphafold2_fixbb_gradient(inputs, config))
    )

    _assert_gradient_result(result, expected_terms=["plddt", "rmsd"], length=length)


@pytest.mark.benchmark("alphafold2-binder-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphafold2_binder_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark the binder protocol: 5 passes of a 118 aa binder against the PD-L1 target (cold + warm).

    The heaviest of the three — the multimer model folds target plus binder each pass — and the
    only one carrying an interface term, so ``i_pae`` joins ``plddt``.
    """
    target = Structure.from_file(_FIXTURE_PDB)
    length = len(target.get_chain_sequence("B"))
    inputs = AlphaFold2BinderGradientInput(
        inputs=[
            AlphaFold2BinderGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits("A" * length, sharpness=2.0)),
                target_pdb=target,
                target_chain="A",
            )
        ],
        temperature=1.0,
    )
    config = AlphaFold2BinderGradientConfig(gradient_terms=["plddt", "i_pae"], num_recycles=1, seed=0)

    result = benchmark_twice(
        request, "alphafold2", lambda: _run_steps(lambda: run_alphafold2_binder_gradient(inputs, config))
    )

    _assert_gradient_result(result, expected_terms=["plddt", "i_pae"], length=length)


@pytest.mark.benchmark("alphafold2-binder-redesign-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphafold2_binder_redesign_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark template redesign: 5 passes redesigning the PD-L1 nanobody chain in place (cold + warm).

    ``plddt`` must come back nonzero here, since ColabDesign's own masks leave it empty under
    redesign; ``dgram_cce`` is the template-agreement term only this protocol and fixbb offer.
    """
    template = Structure.from_file(_FIXTURE_PDB)
    binder = template.get_chain_sequence("B")
    inputs = AlphaFold2BinderRedesignGradientInput(
        inputs=[
            AlphaFold2BinderRedesignGradientCandidate(
                logits=ProteinSequenceLogits(one_hot_protein_logits(binder, sharpness=2.0)),
                template_pdb=template,
                target_chain="A",
                binder_chain="B",
            )
        ],
        temperature=1.0,
    )
    config = AlphaFold2BinderRedesignGradientConfig(gradient_terms=["plddt", "dgram_cce"], num_recycles=1, seed=0)

    result = benchmark_twice(
        request, "alphafold2", lambda: _run_steps(lambda: run_alphafold2_binder_redesign_gradient(inputs, config))
    )

    _assert_gradient_result(result, expected_terms=["plddt", "dgram_cce"], length=len(binder))
    assert result.results[0].terms[0].term_value > 0
