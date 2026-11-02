"""Tests for the CRISPRitz off-target search wrapper."""

from importlib import import_module
from pathlib import Path

import pytest

from proto_tools import (
    CRISPRitzSearchConfig,
    CRISPRitzSearchInput,
    run_crispritz_search,
)
from proto_tools.tools.sequence_alignment.crispritz.standalone.run import (
    _hash_references,
    _parse_hits,
    _split_fasta,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output


def test_input_contract_normalizes_rna_and_rejects_nonuniform_guides() -> None:
    inputs = CRISPRitzSearchInput(guides="ACGU", reference_fasta="reference.fna")
    assert inputs.guides == ["ACGT"]
    with pytest.raises(ValueError, match="same length"):
        CRISPRitzSearchInput(guides=["ACGT", "ACGTA"], reference_fasta="reference.fna")
    with pytest.raises(ValueError, match="non-canonical"):
        CRISPRitzSearchInput(guides=["ACGN"], reference_fasta="reference.fna")


def test_pam_contract_supports_both_orientations() -> None:
    assert CRISPRitzSearchConfig().pam == "NGG"
    assert CRISPRitzSearchConfig(pam="tttv", pam_side="five_prime").pam == "TTTV"
    with pytest.raises(ValueError, match="invalid IUPAC"):
        CRISPRitzSearchConfig(pam="NGZ")


def test_wrapper_dispatches_batch_and_returns_typed_hits(monkeypatch, tmp_path: Path) -> None:
    module = import_module("proto_tools.tools.sequence_alignment.crispritz.crispritz_search")
    captured = {}

    def fake_dispatch(toolkit, payload, **kwargs):
        captured.update(toolkit=toolkit, payload=payload, kwargs=kwargs)
        return {
            "hits": [
                {
                    "guide": "GACTGCATTGCATGACTGCA",
                    "bulge_type": "dna",
                    "crispritz_bulge_type": "DNA",
                    "guide_alignment": "GACTGCATTG-CATGACTGCANNN",
                    "target_alignment": "GACTGCATTGTCATGACTGCAAGG",
                    "chromosome": "chr1",
                    "start": 799,
                    "end": 822,
                    "strand": "+",
                    "mismatches": 0,
                    "bulge_size": 1,
                    "total_edits": 1,
                }
            ],
            "reference_sha256": "a" * 64,
            "index_reused": False,
        }

    monkeypatch.setattr(module.ToolInstance, "dispatch", fake_dispatch)
    reference = tmp_path / "reference.fna"
    reference.write_text(">chr1\nACGT\n")
    inputs = CRISPRitzSearchInput(
        guides=["GACTGCATTGCATGACTGCA", "AAAAAAAAAAAAAAAAAAAA"],
        reference_fasta=str(reference),
        reference_id="GCF_example.1",
    )
    output = run_crispritz_search(inputs, CRISPRitzSearchConfig(max_total_edits=3))

    assert captured["toolkit"] == "crispritz"
    assert captured["payload"]["guides"] == inputs.guides
    assert captured["payload"]["max_total_edits"] == 3
    assert output.hits[0].bulge_type == "dna"
    assert (output.hits[0].start, output.hits[0].end) == (799, 822)
    assert output.reference_id == "GCF_example.1"
    assert output.metadata["coordinate_system"] == "1-indexed-inclusive"


def test_missing_reference_fails_before_tool_dispatch(monkeypatch, tmp_path: Path) -> None:
    module = import_module("proto_tools.tools.sequence_alignment.crispritz.crispritz_search")

    def fail_dispatch(*args, **kwargs):
        raise AssertionError("dispatch must not run")

    monkeypatch.setattr(module.ToolInstance, "dispatch", fail_dispatch)
    inputs = CRISPRitzSearchInput(guides=["ACGT"], reference_fasta=str(tmp_path / "missing.fna"))
    with pytest.raises(FileNotFoundError, match="reference_fasta does not exist"):
        run_crispritz_search(inputs, CRISPRitzSearchConfig())


def test_multirecord_fasta_is_split_without_changing_identifiers(tmp_path: Path) -> None:
    source = tmp_path / "assembly.fna"
    source.write_text(">chr1 description\nACGT\nACGT\n>plasmid\nTTTT\n")
    destination = tmp_path / "records"

    assert _split_fasta([source], destination) == 2
    assert (destination / "record_00000000.fa").read_text() == ">chr1\nACGTACGT\n"
    assert (destination / "record_00000001.fa").read_text() == ">plasmid\nTTTT\n"


def test_reference_checksum_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.fa"
    second = tmp_path / "relocated.fa"
    first.write_text(">chr1\nACGT\n")
    second.write_bytes(first.read_bytes())

    assert _hash_references([first]) == _hash_references([second])


def test_parser_preserves_joint_bulge_type_and_converts_to_1_indexed_coordinates(tmp_path: Path) -> None:
    targets = tmp_path / "result.targets.txt"
    targets.write_text(
        "#Bulge type\tcrRNA\tDNA\tChromosome\tPosition\tCluster Position\tDirection\tMismatches\tBulge Size\tTotal\n"
        "RNA,DNA\tGACTGCATTGCATGACTGCANNN\tGACTGCATTGCATGACTGCAAGG\tchr1\t0\t1\t+\t1\t2\t3\n"
    )

    [hit] = _parse_hits(targets, pam_length=3, pam_side="three_prime")
    assert hit["guide"] == "GACTGCATTGCATGACTGCA"
    assert hit["bulge_type"] == "dna_rna"
    assert hit["crispritz_bulge_type"] == "RNA,DNA"
    assert (hit["start"], hit["end"]) == (1, 23)


@pytest.mark.benchmark("crispritz-search")
@pytest.mark.slow
def test_crispritz_search_benchmark(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Benchmark 100 guides against a synthetic 100-kb local reference index."""
    guide = "GACTGCATTGCATGACTGCA"
    reference = tmp_path / "reference.fasta"
    reference.write_text(">synthetic\n" + ("ACGT" * 24_993) + f"AAAA{guide}TGGAAA\n")
    guides = [guide, *("A" * 19 + base for base in "ACG"), *("C" * 19 + base for base in "AGT")]
    inputs = CRISPRitzSearchInput(guides=guides * 14 + guides[:2], reference_fasta=str(reference))
    config = CRISPRitzSearchConfig(threads=4)

    result = benchmark_twice(request, "crispritz", lambda: run_crispritz_search(inputs, config))

    validate_output(result)
    assert result.tool_id == "crispritz-search"
    assert len(result.guides) == 100
