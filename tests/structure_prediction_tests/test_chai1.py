"""tests/structure_prediction_tests/test_chai1.py.

Benchmark and unit tests for Chai1 structure prediction.

Cross-tool integration coverage lives in ``test_structure_prediction.py``;
this file holds the cold/warm benchmark, Chai-1's glycan-aware token counter,
and its token-budget validator. The shared structure-token counter is tested in
``test_shared_data_models.py``.
"""

import pytest

from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Chai1Config,
    Chai1Input,
    run_chai1,
)
from proto_tools.tools.structure_prediction.chai1.chai1 import run_chai1_on_complex
from proto_tools.tools.structure_prediction.chai1.helpers import count_chai1_tokens
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
)
from tests._structure_fixtures import synthetic_cif
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

_persistent_tool = make_persistent_fixture("chai1")


def test_msa_to_pqt_appends_deep_unpaired_with_empty_pairing_key(tmp_path):
    """Paired rows keep row-index pairing keys; the deep unpaired MSA is appended with empty keys.

    chai_lab pairs rows whose ``pairing_key`` matches across chains and treats empty
    keys as unpaired depth, so one pqt carries both the row-aligned paired rows and
    the deeper per-chain unpaired rows.
    """
    import pandas as pd

    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.chai1.chai1 import _msa_to_pqt_file

    query = "MKTAYIAKQR"
    paired = MSA(aligned_sequences=[query, "MKTAYIAKQA", "MKTAYIAKQE"])  # query + 2 paired rows
    # Deep unpaired: query + 4 rows, 2 of which overlap the paired set (should dedup).
    unpaired = MSA(aligned_sequences=[query, "MKTAYIAKQA", "MKTAYIAKQE", "MKTAYIAKQW", "MKTAYIAKQY"])

    pqt_path = str(tmp_path / "chain.aligned.pqt")
    _msa_to_pqt_file(msa=paired, pqt_path=pqt_path, paired=True, unpaired_msa=unpaired)

    df = pd.read_parquet(pqt_path)
    keys = df["pairing_key"].tolist()
    seqs = df["sequence"].tolist()
    assert len(df) == 5  # query + 2 paired + 2 unique unpaired (overlaps deduped)
    assert keys[0] == "0" and df.iloc[0]["source_database"] == "query"  # query row, keyed as chai-lab's pipeline does
    assert keys[1:3] == ["1", "2"]  # paired rows: row-index pairing keys
    assert keys[3:] == ["", ""]  # appended deep-unpaired rows: empty pairing key
    assert "MKTAYIAKQW" in seqs and "MKTAYIAKQY" in seqs  # new unpaired depth present


def test_msa_to_pqt_without_unpaired_keeps_paired_only(tmp_path):
    """With no unpaired MSA, only the paired rows are written (prior behavior)."""
    import pandas as pd

    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.chai1.chai1 import _msa_to_pqt_file

    paired = MSA(aligned_sequences=["MKTAYIAKQR", "MKTAYIAKQA", "MKTAYIAKQE"])
    pqt_path = str(tmp_path / "chain.aligned.pqt")
    _msa_to_pqt_file(msa=paired, pqt_path=pqt_path, paired=True)

    df = pd.read_parquet(pqt_path)
    assert len(df) == 3
    assert df["pairing_key"].tolist() == ["0", "1", "2"]


@pytest.mark.uses_env
def test_chai1_env_parses_msa_pqt(tmp_path):
    """The chai1 env reads the MSA parquet proto-tools writes (needs a NumPy 1.x-compatible pyarrow)."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.chai1.chai1 import _msa_to_pqt_file
    from proto_tools.utils import run_in_env

    pqt_path = tmp_path / "chain.aligned.pqt"
    _msa_to_pqt_file(msa=MSA(aligned_sequences=["MKTAYIAKQR", "MKTAYIAKQA", "MKTAYIAKQE"]), pqt_path=str(pqt_path))

    code = (
        "import sys\n"
        "from chai_lab.data.parsing.msas.aligned_pqt import parse_aligned_pqt_to_msa_context\n"
        "print(parse_aligned_pqt_to_msa_context(sys.argv[1]).depth)\n"
    )
    output = run_in_env("chai1", code=code, args=[str(pqt_path)], timeout=1800)
    assert output.strip().splitlines()[-1] == "3"


def test_chai1_reports_recycle_msa_subsampling(monkeypatch):
    """Any positive recycle_msa_subsample reports chai-lab's fixed 4096-row per-recycle draw."""
    sequence = "MKTAYIAKQR"
    complex_ = Complex(chains=[Chain(sequence=sequence, entity_type="protein")])
    complex_msas = ComplexMSAs(per_chain={0: MSA(aligned_sequences=[sequence, sequence])})

    def fake_dispatch(*args, **kwargs):
        return {
            "cif_output": synthetic_cif(["A"]),
            "metrics": {
                "avg_plddt": 0.8,
                "ptm": 0.7,
                "iptm": 0.0,
                "avg_pae": 1.0,
                "pae": None,
                "confidence_score": 0.75,
            },
        }

    monkeypatch.setattr("proto_tools.tools.structure_prediction.chai1.chai1.ToolInstance.dispatch", fake_dispatch)
    structure = run_chai1_on_complex(
        complex_,
        Chai1Config(use_msa=True, recycle_msa_subsample=32),
        complex_msas=complex_msas,
    )

    report = structure.msa_conditioning
    assert report.chains[0].status == "consumed"
    assert report.subsampling.mode == "stochastic"
    assert report.subsampling.max_rows == 4096


@pytest.mark.benchmark("chai1-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_chai1_benchmark(request):
    """Benchmark chai1-prediction on the MfnG protein + L-tyrosine ligand with MSA (cold + warm).

    Single ~390-residue protein-ligand complex. The MSA search is included because it is usually
    the dominant cost of a real fold, and because nothing else exercises that path.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = Chai1Input(complexes=[complex_])
    config = Chai1Config(use_msa=True, verbose=True)

    result = benchmark_twice(request, "chai1", lambda: run_chai1(inputs=inputs, config=config))

    assert result.success, "Chai1 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)


# ============================================================================
# Token-budget validation (CPU-only, no model dispatch)
# ============================================================================
def test_count_chai1_tokens_glycan_sums_sugar_heavy_atoms():
    """Glycan sugars add their heavy-atom counts; other chains delegate to the shared counter."""
    chains = [
        Chain(sequence="MKTL", entity_type="protein"),
        Chain(sequence="MAN(6-1 FUC)(4-1 MAN)", entity_type="glycan"),
    ]
    assert count_chai1_tokens(chains) == 4 + 12 + 11 + 12  # protein 4 + MAN 12 + FUC 11 + MAN 12


@pytest.mark.uses_env
def test_chai1_reported_msa_caps_match_chai_lab():
    """chai-lab's per-recycle draw and uniref90 quota are the rows the report claims."""
    from proto_tools.tools.structure_prediction.chai1.chai1 import (
        CHAI1_RECYCLE_SUBSAMPLE_ROWS,
        CHAI1_UNIREF90_QUOTA_ROWS,
    )
    from proto_tools.utils import run_in_env

    code = (
        "import inspect\n"
        "from chai_lab.data.dataset.msas.utils import subsample_and_reorder_msa_feats_n_mask as subsample\n"
        "from chai_lab.data.parsing.msas.data_source import MSADataSource, msa_dataset_source_to_quota\n"
        "print(inspect.signature(subsample).parameters['select_n_rows'].default)\n"
        "print(msa_dataset_source_to_quota[MSADataSource.UNIREF90])\n"
    )
    select_n_rows, quota = run_in_env("chai1", code=code, timeout=1800).strip().splitlines()[-2:]
    assert (int(select_n_rows), int(quota)) == (CHAI1_RECYCLE_SUBSAMPLE_ROWS, CHAI1_UNIREF90_QUOTA_ROWS)
