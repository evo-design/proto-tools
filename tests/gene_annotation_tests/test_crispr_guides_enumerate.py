"""Tests for deterministic PAM-aware CRISPR guide enumeration."""

import hashlib
import time

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    CRISPRGuideEnumerationConfig,
    CRISPRGuideEnumerationInput,
    CRISPRGuideRegion,
    run_crispr_guides_enumerate,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

_GUIDE = "GACTGCATTGCATGACTGCA"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_IUPAC_EXPECTED = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def test_regions_normalize_and_sequence_checksum_is_verified() -> None:
    sequence = f"AAAA{_GUIDE}TGGAAA"
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    inputs = CRISPRGuideEnumerationInput(
        regions=CRISPRGuideRegion(
            sequence_id="chr1", sequence=sequence.lower(), expected_sequence_sha256=digest.upper()
        )
    )

    assert len(inputs.regions) == 1
    assert inputs.regions[0].sequence == sequence
    assert inputs.regions[0].expected_sequence_sha256 == digest

    with pytest.raises(ValidationError, match="does not match"):
        CRISPRGuideRegion(sequence_id="chr1", sequence=sequence, expected_sequence_sha256="0" * 64)


def test_three_prime_pam_plus_strand_coordinates_and_rs3_context() -> None:
    sequence = f"AAAA{_GUIDE}TGGAAA"
    result = run_crispr_guides_enumerate(
        CRISPRGuideEnumerationInput(
            regions=CRISPRGuideRegion(sequence_id="chr7", sequence=sequence, start=101, reference_id="GCF_test.1")
        ),
        CRISPRGuideEnumerationConfig(strands="plus"),
    )

    assert result.success
    assert len(result.results) == 1
    candidate = result.results[0].candidates[0]
    assert len(result.results[0].candidates) == 1
    assert candidate.protospacer == _GUIDE
    assert candidate.pam == "TGG"
    assert candidate.pam_strand == "+"
    assert candidate.target_strand == "-"
    assert (candidate.protospacer_start, candidate.protospacer_end) == (105, 124)
    assert (candidate.pam_start, candidate.pam_end) == (125, 127)
    assert (candidate.context_start, candidate.context_end) == (101, 130)
    assert candidate.context_sequence == sequence
    assert candidate.context_complete
    assert candidate.context_is_canonical
    assert len(candidate.context_sequence) == 30
    assert result.metadata["selection_applied"] is False
    assert result.metadata["scoring_applied"] is False


def test_three_prime_pam_minus_strand_coordinates() -> None:
    oriented = f"AAAA{_GUIDE}TGGAAA"
    result = run_crispr_guides_enumerate(
        CRISPRGuideEnumerationInput(
            regions=CRISPRGuideRegion(sequence_id="chr7", sequence=_reverse_complement(oriented), start=101)
        ),
        CRISPRGuideEnumerationConfig(strands="minus"),
    )

    candidate = result.results[0].candidates[0]
    assert len(result.results[0].candidates) == 1
    assert candidate.protospacer == _GUIDE
    assert candidate.pam_strand == "-"
    assert candidate.target_strand == "+"
    assert (candidate.protospacer_start, candidate.protospacer_end) == (107, 126)
    assert (candidate.pam_start, candidate.pam_end) == (104, 106)
    assert (candidate.context_start, candidate.context_end) == (101, 130)
    assert candidate.context_sequence == oriented


def test_five_prime_pam_and_partial_context() -> None:
    sequence = f"TTTA{_GUIDE}"
    result = run_crispr_guides_enumerate(
        CRISPRGuideEnumerationInput(regions=CRISPRGuideRegion(sequence_id="contig", sequence=sequence, start=11)),
        CRISPRGuideEnumerationConfig(pam="TTTV", pam_side="five_prime", strands="plus", context_5p=3, context_3p=2),
    )

    candidate = result.results[0].candidates[0]
    assert candidate.pam == "TTTA"
    assert (candidate.pam_start, candidate.pam_end) == (11, 14)
    assert (candidate.protospacer_start, candidate.protospacer_end) == (15, 34)
    assert candidate.context_sequence == sequence
    assert not candidate.context_complete
    assert candidate.context_5p_length == 0
    assert candidate.context_3p_length == 0


@pytest.mark.parametrize(("pam", "expected_bases"), _IUPAC_EXPECTED.items())
def test_iupac_pam_grammar(pam: str, expected_bases: str) -> None:
    observed = set()
    for base in "ACGT":
        result = run_crispr_guides_enumerate(
            CRISPRGuideEnumerationInput(regions=CRISPRGuideRegion(sequence_id="test", sequence=f"A{base}")),
            CRISPRGuideEnumerationConfig(
                pam=pam, pam_side="three_prime", protospacer_length=1, strands="plus", context_5p=0, context_3p=0
            ),
        )
        if result.results[0].candidates:
            observed.add(base)
    assert observed == set(expected_bases)


def test_ambiguous_target_windows_are_not_emitted() -> None:
    sequence = f"AAAA{_GUIDE[:-1]}NTGGAAA"
    result = run_crispr_guides_enumerate(
        CRISPRGuideEnumerationInput(regions=CRISPRGuideRegion(sequence_id="chr1", sequence=sequence)),
        CRISPRGuideEnumerationConfig(strands="plus"),
    )

    assert result.results[0].candidates == []
    assert result.results[0].ambiguous_windows_skipped > 0


def test_ambiguous_exterior_context_is_annotated_without_removing_candidate() -> None:
    sequence = f"NAAA{_GUIDE}TGGAAA"
    result = run_crispr_guides_enumerate(
        CRISPRGuideEnumerationInput(regions=CRISPRGuideRegion(sequence_id="chr1", sequence=sequence)),
        CRISPRGuideEnumerationConfig(strands="plus"),
    )

    assert len(result.results[0].candidates) == 1
    assert result.results[0].candidates[0].context_complete
    assert not result.results[0].candidates[0].context_is_canonical


def test_batch_cardinality_and_csv_export(tmp_path) -> None:
    sequence = f"AAAA{_GUIDE}TGGAAA"
    inputs = CRISPRGuideEnumerationInput(
        regions=[
            CRISPRGuideRegion(sequence_id="first", sequence=sequence),
            CRISPRGuideRegion(sequence_id="second", sequence="A" * len(sequence)),
        ]
    )
    result = run_crispr_guides_enumerate(inputs, CRISPRGuideEnumerationConfig(strands="plus"))

    assert [item.sequence_id for item in result.results] == ["first", "second"]
    assert [len(item.candidates) for item in result.results] == [1, 0]
    result.export("guides", tmp_path, file_format="csv")
    rows = (tmp_path / "guides.csv").read_text().splitlines()
    assert len(rows) == 2
    assert rows[1].startswith("first,")


@pytest.mark.benchmark("crispr-guides-enumerate")
@pytest.mark.slow
def test_crispr_guides_enumerate_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark 100 independently identified 10-kb reference intervals."""
    sequence = ("ACGT" * 2494) + f"AAAA{_GUIDE}TGGAAA" + ("ACGT" * 2)
    inputs = CRISPRGuideEnumerationInput(
        regions=[CRISPRGuideRegion(sequence_id=f"region_{index}", sequence=sequence) for index in range(100)]
    )
    config = CRISPRGuideEnumerationConfig(strands="both")

    start = time.perf_counter()
    result = run_crispr_guides_enumerate(inputs, config)
    elapsed = time.perf_counter() - start
    request.node.user_properties.append(("seconds", elapsed))

    validate_output(result)
    assert result.tool_id == "crispr-guides-enumerate"
    assert len(result.results) == 100
    assert all(item.candidates for item in result.results)
