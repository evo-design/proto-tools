"""Tests that every MSA-taking tool uses the MSA rows it is given, for every bundle shape."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.shared_data_models import Chain, Complex, ComplexMSAs
from proto_tools.tools.tool_registry import ToolRegistry
from tests._structure_fixtures import single_chain_pdb, synthetic_cif

# Each registered tool whose input takes ``msas`` runs end-to-end with its worker stubbed, once per
# bundle shape, and its ``msa_conditioning`` report is checked: a chain given homolog rows is
# ``consumed`` unless the backend has a declared limitation, and a chain with only unpaired rows is
# never reported as paired. A new MSA-taking tool fails ``test_every_msa_tool_is_covered`` until it
# gets a stub in ``_STUBS``, so it is checked against every shape from its first commit.

_A, _B, _C = "MKTAYIAKQR", "GSHMSLFDFF", "PEELLKKLEE"


def _rows(query: str, depth: int) -> MSA:
    """A query followed by ``depth - 1`` distinct homolog rows (one substitution each)."""
    substitutes = "WYV"
    return MSA(aligned_sequences=[query] + [substitutes[i] + query[1:] for i in range(depth - 1)])


@dataclass(frozen=True)
class _Scenario:
    sequences: tuple[str, ...]
    msas: ComplexMSAs


_SCENARIOS = {
    "monomer": _Scenario((_A,), ComplexMSAs(per_chain={0: _rows(_A, 3)})),
    "heteromer_unpaired": _Scenario((_A, _B), ComplexMSAs(per_chain={0: _rows(_A, 3), 1: _rows(_B, 2)})),
    "heteromer_paired_with_depth": _Scenario(
        (_A, _B),
        ComplexMSAs(
            per_chain={0: _rows(_A, 2), 1: _rows(_B, 2)},
            paired=True,
            unpaired_per_chain={0: _rows(_A, 3), 1: _rows(_B, 3)},
        ),
    ),
    # Chains A and B are paired; chain C has unpaired rows but no primary alignment (issue #349).
    "heteromer_unpaired_only_chain": _Scenario(
        (_A, _B, _C),
        ComplexMSAs(
            per_chain={0: _rows(_A, 2), 1: _rows(_B, 2)},
            paired=True,
            unpaired_per_chain={0: _rows(_A, 3), 2: _rows(_C, 3)},
        ),
    ),
    # One paired chain and one with only unpaired rows: the paired rows have no partner chain.
    "dimer_paired_beside_unpaired_only": _Scenario(
        (_A, _B),
        ComplexMSAs(per_chain={0: _rows(_A, 2)}, paired=True, unpaired_per_chain={1: _rows(_B, 3)}),
    ),
    "homodimer": _Scenario((_A, _A), ComplexMSAs(per_chain={0: _rows(_A, 3), 1: _rows(_A, 3)})),
    # AF3-lineage backends (AF3, Protenix, OpenDDE) do not pair identical chains; the others do.
    "homodimer_paired": _Scenario((_A, _A), ComplexMSAs(per_chain={0: _rows(_A, 3), 1: _rows(_A, 3)}, paired=True)),
    "heteromer_one_chain_without_msa": _Scenario((_A, _B), ComplexMSAs(per_chain={0: _rows(_A, 3)})),
}


# ── Worker stubs ─────────────────────────────────────────────────────────────

Reply = Callable[[dict[str, Any], list[str]], dict[str, Any]]


def _chain_ids(sequences: list[str]) -> list[str]:
    return [chr(ord("A") + index) for index in range(len(sequences))]


@dataclass(frozen=True)
class _Stub:
    """How to run one tool without its worker.

    Attributes:
        reply: Builds the worker's reply from the dispatched payload and the protein sequences.
        config: Config overrides that select the tool's MSA-consuming mode.
        ligand: Whether each complex needs a ligand (the tool rejects protein-only complexes).
    """

    reply: Reply
    config: dict[str, Any] = field(default_factory=dict)
    ligand: bool = False


def _cif_reply(cif_key: str, metrics: dict[str, Any], metrics_key: str = "metrics") -> Reply:
    return lambda _payload, sequences: {cif_key: synthetic_cif(_chain_ids(sequences)), metrics_key: metrics}


def _alphafold3_reply(payload: dict[str, Any], sequences: list[str]) -> dict[str, Any]:
    path = Path(payload["output_dir"]) / "stub.cif"
    path.write_text(synthetic_cif(_chain_ids(sequences)))
    return {"structure_cif": str(path), "metrics": {"avg_plddt": 0.9, "ptm": 0.8}}


def _esmfold_reply(_payload: dict[str, Any], sequences: list[str]) -> dict[str, Any]:
    # ESMFold folds the chains linked into one, then splits the result back by chain length.
    return {
        "results": [{"pdb": single_chain_pdb(sum(map(len, sequences))), "avg_plddt": 0.8, "ptm": 0.5, "avg_pae": 6.0}]
    }


_STUBS: dict[str, _Stub] = {
    "alphafold2-prediction": _Stub(
        lambda _payload, sequences: {
            "pdb": synthetic_cif(_chain_ids(sequences)),
            "avg_plddt": 0.85,
            "ptm": 0.72,
            "iptm": None,
            "avg_pae": 1.5,
            "pae": None,
        }
    ),
    "alphafold3-prediction": _Stub(_alphafold3_reply),
    "boltz2-prediction": _Stub(
        _cif_reply(
            "structure_cif_output",
            {
                "confidence_score": 0.8,
                "ptm": 0.7,
                "iptm": 0.0,
                "chains_ptm": [],
                "pair_chains_iptm": [],
                "avg_pae": 1.0,
                "pae": None,
            },
        )
    ),
    "boltz2-affinity": _Stub(
        _cif_reply(
            "structure_cif_output",
            {"affinity_pred_value": -1.2, "affinity_probability_binary": 0.75},
            metrics_key="affinity_metrics",
        ),
        ligand=True,
    ),
    "chai1-prediction": _Stub(
        _cif_reply(
            "cif_output",
            {"avg_plddt": 0.8, "ptm": 0.7, "iptm": 0.0, "avg_pae": 1.0, "pae": None, "confidence_score": 0.7},
        )
    ),
    "esmfold-prediction": _Stub(_esmfold_reply),
    "esmfold2-prediction": _Stub(
        _cif_reply("structure_cif_output", {"plddt": 0.8, "ptm": 0.7, "avg_pae": 1.0}),
        config={"model_checkpoint": "esmfold2"},
    ),
    "mol1-prediction": _Stub(
        _cif_reply(
            "structure_cif_output",
            {"avg_plddt": 91.2, "ptm": 0.85, "iptm": 0.0, "gpde": 1.2, "ranking_score": 0.17, "has_clash": False},
        )
    ),
    "opendde-prediction": _Stub(
        _cif_reply(
            "structure_cif_output",
            {"avg_plddt": 91.2, "ptm": 0.85, "iptm": 0.0, "gpde": 1.2, "ranking_score": 0.77, "has_clash": False},
        )
    ),
    "protenix-prediction": _Stub(
        lambda _payload, sequences: {
            "results": [
                {"structure_cif_output": synthetic_cif(_chain_ids(sequences)), "metrics": {"ranking_score": 0.8}}
            ]
        }
    ),
    "rf3-prediction": _Stub(
        _cif_reply(
            "structure_cif_output",
            {
                "avg_plddt": 0.8,
                "ptm": 0.7,
                "avg_pae": 1.0,
                "pde": 1.0,
                "ranking_score": 0.7,
                "chain_ptm": [],
                "chain_pair_pae": [],
                "chain_pair_pae_min": [],
                "chain_pair_pde": [],
                "chain_pair_pde_min": [],
                "has_clash": False,
            },
        )
    ),
}

# MSA-taking tools that build no MSA conditioning report, and why they are not checked here.
_EXEMPT = {
    "bioemu-sample": "samples single-chain ensembles and attaches no msa_conditioning report",
}

# Backends whose upstream pairs identical chains' rows with each other. AF3, Protenix and OpenDDE
# pair only across distinct protein sequences (need_msa_pairing), as does Mol-1, and RF3 treats a homomer as unpaired.
_PAIRS_IDENTICAL_CHAINS = {"boltz2-prediction", "boltz2-affinity", "chai1-prediction", "esmfold2-prediction"}

# Ignored reasons that are genuine backend limitations rather than dropped rows.
_BACKEND_LIMITS = {
    "esmfold-prediction": "ESMFold is a single-sequence model",
}


def _msa_tool_keys() -> set[str]:
    return {spec.key for spec in ToolRegistry.list_all() if "msas" in spec.input_model.model_fields}


def test_every_msa_tool_is_covered():
    """A new MSA-taking tool needs a worker stub here (or a documented exemption)."""
    missing = _msa_tool_keys() - set(_STUBS) - set(_EXEMPT)
    assert not missing, f"add a _STUBS entry for {sorted(missing)} so its MSA handling is checked"
    stale = (set(_STUBS) | set(_EXEMPT)) - _msa_tool_keys()
    assert not stale, f"{sorted(stale)} no longer take MSAs; drop their entries"


@pytest.mark.parametrize("scenario_name", _SCENARIOS)
@pytest.mark.parametrize("tool_key", sorted(_STUBS))
def test_supplied_msa_rows_reach_the_backend(tool_key, scenario_name, monkeypatch):
    """Chains given homolog rows are consumed, unpaired-only chains never pair, and pairing needs a partner."""
    stub, scenario = _STUBS[tool_key], _SCENARIOS[scenario_name]
    spec = ToolRegistry.get(tool_key)
    chains: list[Chain | Fragment] = [Chain(sequence=seq, entity_type="protein") for seq in scenario.sequences]
    if stub.ligand:
        chains.append(Fragment(ccd_code="ATP"))

    monkeypatch.setattr(
        "proto_tools.utils.ToolInstance.dispatch",
        lambda _toolkit, payload, **_kwargs: stub.reply(payload, list(scenario.sequences)),
    )
    output = spec.function(
        spec.input_model(complexes=[Complex(chains=chains)], msas=[scenario.msas]),
        spec.config_model(device="cpu", **stub.config),
    )

    msas = scenario.msas
    for chain in output.structures[0].msa_conditioning.chains:
        supplied = [
            msa
            for msa in (msas.per_chain.get(chain.chain_index), (msas.unpaired_per_chain or {}).get(chain.chain_index))
            if msa
        ]
        where = f"{tool_key} / {scenario_name} / chain {chain.chain_id}"
        if not supplied:
            assert chain.status == "not_requested", where
            continue
        if chain.status != "consumed":
            limit = _BACKEND_LIMITS.get(tool_key)
            assert limit is not None and chain.ignored_reason == limit, (
                f"{where}: supplied rows were {chain.status} ({chain.ignored_reason})"
            )
        if chain.chain_index not in msas.per_chain:
            assert chain.mode != "paired", f"{where}: a chain with only unpaired rows was reported paired"
    modes = [chain.mode for chain in output.structures[0].msa_conditioning.chains]
    assert modes.count("paired") != 1, f"{tool_key} / {scenario_name}: one chain reported paired with no partner"
    if scenario_name == "homodimer_paired":
        if tool_key in _PAIRS_IDENTICAL_CHAINS:
            assert modes == ["paired", "paired"], f"{tool_key}: paired homodimer reported {modes}"
        else:
            assert "paired" not in modes, f"{tool_key}: paired homodimer reported {modes}"
