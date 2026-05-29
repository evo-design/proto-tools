"""tests/structure_dynamics_tests/test_bioemu.py.

Tests for BioEmu.
"""

from unittest.mock import patch

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_dynamics.bioemu import (
    BioEmuConfig,
    BioEmuInput,
    run_bioemu,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
)
from proto_tools.utils import ToolInstance
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_SAMPLE_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

_SAMPLE_PDB_CONTENT = (
    "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C\n"
    "END\n"
)


# ── Input validation ─────────────────────────────────────────────────────────


def test_input_rejects_multi_chain_complex():
    with pytest.raises(ValueError, match="single-chain"):
        BioEmuInput(
            complexes=[
                Complex(
                    chains=[
                        {"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"},
                        {"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"},
                    ]
                )
            ]
        )


def test_input_rejects_non_protein_entity():
    with pytest.raises(ValueError, match="only supports: protein"):
        BioEmuInput(complexes=[Complex(chains=[{"sequence": "ACGT", "entity_type": "dna"}])])


def test_input_rejects_invalid_amino_acids():
    with (
        patch(
            "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value={"1", "2", "3"},
        ),
        pytest.raises(ValueError, match="Invalid protein characters"),
    ):
        BioEmuInput(
            complexes=[
                Complex(
                    chains=[
                        {
                            "sequence": "MVLSPADKTNVKAAW123",
                            "entity_type": "protein",
                        }
                    ]
                )
            ]
        )


def test_input_warns_on_long_sequence(caplog):
    long_sequence = "A" * 600
    with (
        patch(
            "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value=set(),
        ),
        caplog.at_level("WARNING"),
    ):
        BioEmuInput(
            complexes=[
                Complex(
                    chains=[
                        {
                            "sequence": long_sequence,
                            "entity_type": "protein",
                        }
                    ]
                )
            ]
        )
    assert "500 residues" in caplog.text


# ── Config validation ────────────────────────────────────────────────────────


def test_config_rejects_num_samples_zero():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BioEmuConfig(num_samples=0)


def test_config_rejects_batch_size_zero():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BioEmuConfig(batch_size=0)


def test_config_rejects_invalid_model_name():
    with pytest.raises(ValueError, match="Input should be"):
        BioEmuConfig(model_name="invalid-model")


def test_config_passes_new_fields_to_dispatch():
    """All five new Config fields must flow to the dispatch payload."""
    # Pre-supply an empty per-complex MSA entry so preprocess() skips ColabFold.
    complex_ = Complex(chains=[{"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"}])
    bioemu_input = BioEmuInput(complexes=[complex_], msas=[{}])
    bioemu_config = BioEmuConfig(
        num_samples=2,
        model_name="bioemu-v1.2",
        denoiser_type="heun",
        denoiser_config="steering.yaml",
        msa_host_url="https://msa.example.com",
        cache_embeds_dir="embeds_cache",
        cache_so3_dir="so3_cache",
    )

    with patch(
        "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
    ) as mock_cls:
        mock_cls.dispatch.return_value = {
            "results": [
                {
                    "pdb_frames": [_SAMPLE_PDB_CONTENT],
                    "num_frames": 1,
                    "num_residues": len(_SAMPLE_SEQUENCE),
                }
            ]
        }
        run_bioemu(bioemu_input, bioemu_config)

    payload = mock_cls.dispatch.call_args[0][1]
    assert payload["model_name"] == "bioemu-v1.2"
    assert payload["denoiser_type"] == "heun"
    assert payload["denoiser_config"] == "steering.yaml"
    assert payload["msa_host_url"] == "https://msa.example.com"
    assert payload["cache_embeds_dir"] == "embeds_cache"
    assert payload["cache_so3_dir"] == "so3_cache"


def test_config_cache_key_invariants():
    """Cache dirs / output_dir excluded; msa_host_url INCLUDED (server change → different MSAs)."""
    base = BioEmuConfig().cache_key()

    # Excluded — storage knobs don't affect the returned ensembles
    assert (
        BioEmuConfig(
            cache_embeds_dir="embeds",
            cache_so3_dir="so3",
            output_dir="raw",
        ).cache_key()
        == base
    )

    # Included — different MSA server can produce different MSAs → different ensembles
    assert BioEmuConfig(msa_host_url="https://other.example.com").cache_key() != base


# ── Output assembly (mocked dispatch) ────────────────────────────────────────


@pytest.mark.slow
def test_multiple_complexes_produce_separate_ensembles():
    complex_ = Complex(chains=[{"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"}])
    bioemu_input = BioEmuInput(complexes=[complex_, complex_])
    bioemu_config = BioEmuConfig(num_samples=10, verbose=False)

    with patch(
        "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
    ) as mock_cls:
        mock_cls.dispatch.return_value = {
            "results": [
                {
                    "pdb_frames": [_SAMPLE_PDB_CONTENT] * 3,
                    "num_frames": 3,
                    "num_residues": len(_SAMPLE_SEQUENCE),
                },
                {
                    "pdb_frames": [_SAMPLE_PDB_CONTENT] * 7,
                    "num_frames": 7,
                    "num_residues": len(_SAMPLE_SEQUENCE),
                },
            ]
        }
        result = run_bioemu(bioemu_input, bioemu_config)

    assert len(result.ensembles) == 2
    assert len(result.ensembles[0].structures) == 3
    assert len(result.ensembles[1].structures) == 7
    assert result.metadata["num_complexes"] == 2
    assert result.metadata["total_structures"] == 10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_bioemu_sample_end_to_end():
    """Test BioEmu conformational ensemble sampling end-to-end."""
    sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
    inputs = BioEmuInput(complexes=[Complex(chains=[{"sequence": sequence, "entity_type": "protein"}])])
    config = BioEmuConfig(num_samples=5, verbose=False)

    result = run_bioemu(inputs, config)
    validate_output(result)

    assert result.tool_id == "bioemu-sample"
    assert len(result.ensembles) == 1
    assert len(result.ensembles[0].structures) >= 1
    assert result.metadata["num_complexes"] == 1
    assert result.metadata["model_name"] == "bioemu-v1.1"

    for structure in result.ensembles[0].structures:
        assert isinstance(structure, Structure)
        assert structure.structure_pdb is not None
        assert len(structure.structure_pdb) > 0


@pytest.mark.uses_gpu
def test_bioemu_holds_score_model_across_calls():
    """Two calls inside persist_tool reuse the held score_model (no on-disk reload)."""
    sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
    inputs = BioEmuInput(complexes=[Complex(chains=[{"sequence": sequence, "entity_type": "protein"}])])
    config = BioEmuConfig(num_samples=2, batch_size=2, seed=0, verbose=False)

    # Echo reload_on_change fields so the framework doesn't restart the worker.
    introspect_payload = {"operation": "introspect_loaded", "model_name": config.model_name}

    with ToolInstance.persist_tool("bioemu"):
        run_bioemu(inputs, config)
        info1 = ToolInstance.dispatch("bioemu", introspect_payload, config=config)

        run_bioemu(inputs, config)
        info2 = ToolInstance.dispatch("bioemu", introspect_payload, config=config)

    assert info1["loaded"] and info2["loaded"]
    assert info1["model_name"] == info2["model_name"] == "bioemu-v1.1"
    # Same Python object across calls — proves the score_model wasn't recreated.
    assert info1["model_id"] is not None
    assert info1["model_id"] == info2["model_id"]


# ── Benchmark ─────────────────────────────────────────────────────────────────

# Barnase (110 aa) — well-characterized small enzyme; comfortably below the
# 500-aa BioEmu performance warning while large enough to exercise the
# diffusion sampler at realistic batched scale.
_BARNASE_SEQUENCE = (
    "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
)


@pytest.mark.benchmark("bioemu-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_bioemu_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark bioemu-sample: 20 conformations of barnase (110 aa), batch_size=10 (cold + warm)."""
    inputs = BioEmuInput(complexes=[Complex(chains=[{"sequence": _BARNASE_SEQUENCE, "entity_type": "protein"}])])
    config = BioEmuConfig(num_samples=20, batch_size=10, verbose=False)

    result = benchmark_twice(request, "bioemu", lambda: run_bioemu(inputs, config))
    validate_output(result)

    assert result.tool_id == "bioemu-sample"
    assert len(result.ensembles) == 1
    assert len(result.ensembles[0].structures) >= 1
    for structure in result.ensembles[0].structures:
        assert structure.structure_pdb is not None
