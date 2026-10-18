"""tests/epitope_prediction_tests/test_netmhcpan.py.

Tests for the netMHCpan MHC class I binding prediction toolkit.

Allele normalization, peptide/protein validation, the derived propensity and summary
metrics, and the stdout parser all run without the binary; the execution tests are
marked ``integration`` and skip cleanly when no netMHCpan install is resolvable.

netMHCpan's academic license forbids publishing benchmark results to third parties, so
nothing here commits a table of real scores or a captured output file as an oracle. The
parser is fed hand-written tables, and the integration tests assert relational and
structural properties, which is also the only version-stable thing to assert: 4.1 and
4.2 are not numerically comparable.
"""

import importlib
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

import proto_tools.utils.standalone_helpers_source as _shs
from proto_tools.tools import (
    NetMHCpanBinding,
    NetMHCpanPeptidesConfig,
    NetMHCpanPeptidesInput,
    NetMHCpanProteinConfig,
    NetMHCpanProteinInput,
    normalize_allele,
    normalize_alleles,
    run_netmhcpan_peptides,
    run_netmhcpan_protein,
)
from proto_tools.tools.epitope_prediction.netmhcpan.shared_data_models import (
    binding_from_record,
    binding_propensity,
    classify,
    summarize,
    validate_peptide,
    validate_protein,
)
from tests.conftest import benchmark_twice, random_protein_sequences

# The standalone run.py imports the worker-injected ``standalone_helpers`` package, which
# only sits on sys.path inside a tool venv. Add its source dir so ``parse_output`` and the
# install resolver can be imported for the parser tests and the integration skip guard.
sys.path.insert(0, str(Path(next(iter(_shs.__path__)))))
_standalone_run = importlib.import_module("proto_tools.tools.epitope_prediction.netmhcpan.standalone.run")

# ── Constants ────────────────────────────────────────────────────────────────

# Literature epitopes used only as loose sanity bounds, never as numeric oracles.
# CMV pp65 495-503, the canonical HLA-A*02:01 epitope.
_A0201_BINDER = "NLVPMVATV"
# EBV EBNA-3A, an HLA-B*07:02-restricted epitope and not A2-restricted.
_A0201_NON_BINDER = "RPHERNGFTVL"

# Influenza A hemagglutinin N-terminal fragment; 36 residues, so 28 nonamer windows.
_PROTEIN = "MKTIIALSYIFCLVFAQKLPGNDNSTATLCLGHHAV"
_WINDOW_LENGTH = 9

# Hand-written stand-ins for netMHCpan's long-format table. The numbers are invented and
# the peptides are synthetic: the parser only cares about the shape, and committing real
# scores would both publish benchmark results and pin the suite to one netMHCpan version.
_RULE = "-" * 60
_EL_HEADER = (
    " Pos         MHC        Peptide      Core Of Gp Gl Ip Il        Icore        Identity  Score_EL %Rank_EL BindLevel"
)
_BA_HEADER = (
    " Pos         MHC        Peptide      Core Of Gp Gl Ip Il        Icore"
    "        Identity  Score_EL %Rank_EL Score_BA %Rank_BA  Aff(nM) BindLevel"
)
_EXP_HEADER = (
    " Pos         MHC        Peptide      Core Of Gp Gl Ip Il        Icore"
    "        Identity  Score_EL %Rank_EL  Exp BindLevel"
)
_EL_ROW_SB = (
    "   1 HLA-A*02:01      AAAAAAAAV AAAAAAAAV  0  0  0  0  0    AAAAAAAAV         PEPLIST 0.9000000    0.100 <= SB"
)
_EL_ROW_NO_LEVEL = (
    "   1 HLA-A*02:01      DDDDDDDDD DDDDDDDDD  0  0  0  0  0    DDDDDDDDD         PEPLIST 0.0010000   40.000"
)
_BA_ROW_SB = (
    "   1 HLA-A*02:01      AAAAAAAAV AAAAAAAAV  0  0  0  0  0    AAAAAAAAV"
    "         PEPLIST 0.9000000    0.100 0.700000    0.300    20.00 <= SB"
)
_EXP_ROW_WB = (
    "   1 HLA-A*02:01      AAAAAAAAV AAAAAAAAV  0  0  0  0  0    AAAAAAAAV"
    "         PEPLIST 0.4000000    1.500 0.800 <= WB"
)


def _table(header, *rows):
    """Wrap hand-written rows in the version banner and horizontal rules netMHCpan prints."""
    return "\n".join(["# NetMHCpan version 4.1b", _RULE, header, _RULE, *rows, _RULE, ""])


def _binding(rank, peptide="AAAAAAAAV", allele="HLA-A02:01", weak_rank=2.0):
    """Build a binding record at a given EL %rank, classified against the 0.5/weak split."""
    return NetMHCpanBinding(
        allele=allele,
        peptide=peptide,
        position=1,
        core=peptide,
        icore=peptide,
        score_el=0.5,
        rank_el=rank,
        binder_class=classify(rank, 0.5, weak_rank),
    )


# ── Allele normalization ─────────────────────────────────────────────────────


@pytest.mark.parametrize("spelling", ["HLA-A02:01", "HLA-A*02:01", "A*02:01"])
def test_normalize_allele_spellings_converge(spelling):
    """All three common spellings converge on the unstarred form the binary accepts."""
    assert normalize_allele(spelling) == "HLA-A02:01"


def test_normalize_allele_leaves_non_human_prefix_untouched():
    """A non-human prefix is passed through verbatim; MHC_pseudo.dat is case-sensitive."""
    assert normalize_allele("BoLA-100901") == "BoLA-100901"


@pytest.mark.parametrize("name", ["", "   "])
def test_normalize_allele_rejects_empty(name):
    """An empty or whitespace-only allele name is rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_allele(name)


def test_normalize_allele_rejects_internal_whitespace():
    """Whitespace inside an allele name is rejected rather than silently stripped."""
    with pytest.raises(ValueError, match="must not contain whitespace"):
        normalize_allele("HLA-A 02:01")


def test_normalize_alleles_dedupes_preserving_order():
    """A panel is de-duplicated after normalization, keeping first-seen order."""
    panel = ["HLA-B07:02", "HLA-A*02:01", "A*02:01", "HLA-B*07:02"]
    assert normalize_alleles(panel) == ["HLA-B07:02", "HLA-A02:01"]


def test_normalize_alleles_rejects_empty_panel():
    """An empty allele panel is rejected; there is nothing to predict against."""
    with pytest.raises(ValueError, match="at least one MHC allele"):
        normalize_alleles([])


def test_config_normalizes_allele_panel():
    """Config normalization runs on the panel, so mixed spellings collapse to one allele."""
    config = NetMHCpanPeptidesConfig(alleles=["A*02:01", "HLA-A02:01"])
    assert config.alleles == ["HLA-A02:01"]


# ── Peptide and protein validation ───────────────────────────────────────────


def test_peptide_input_single_string_normalizes_to_list():
    """A single peptide string is normalized to a one-element list."""
    assert NetMHCpanPeptidesInput(peptides=_A0201_BINDER).peptides == [_A0201_BINDER]


def test_peptide_input_rejects_off_alphabet_explaining_x_substitution():
    """Off-alphabet residues are rejected, and the message names the X-substitution hazard."""
    with pytest.raises(ValueError, match="silently rewrites these to X"):
        NetMHCpanPeptidesInput(peptides=["NLVPMVATB"])


def test_peptide_input_rejects_lowercase():
    """Lowercase input is rejected rather than up-cased; netMHCpan would rewrite it to X."""
    with pytest.raises(ValueError, match="non-standard residue"):
        NetMHCpanPeptidesInput(peptides=[_A0201_BINDER.lower()])


@pytest.mark.parametrize("peptide", ["NLVPMVA", "NLVPMVATVNLVPMV"])
def test_peptide_input_rejects_out_of_range_length(peptide):
    """Peptides shorter than 8 or longer than 14 residues are rejected."""
    with pytest.raises(ValueError, match="must be 8-14 residues"):
        NetMHCpanPeptidesInput(peptides=[peptide])


def test_validate_peptide_error_names_the_offending_index():
    """The validator reports which list position failed, not just that something did."""
    with pytest.raises(ValueError, match=r"peptides\[3\]"):
        validate_peptide("NLVPMVATB", 3)


def test_protein_input_rejects_off_alphabet():
    """Off-alphabet residues in a protein are rejected with the same X-substitution reason."""
    with pytest.raises(ValueError, match="silently rewrites these to X"):
        NetMHCpanProteinInput(sequences=["MKTIIALSYXFCLVFA"])


def test_protein_input_rejects_sequence_shorter_than_min_peptide():
    """A sequence shorter than the shortest class I peptide has no window to score."""
    with pytest.raises(ValueError, match="shorter than the shortest"):
        NetMHCpanProteinInput(sequences=["MKTIIAL"])


def test_validate_protein_accepts_minimum_length_sequence():
    """An 8-mer is exactly long enough to window and is accepted unchanged."""
    assert validate_protein("MKTIIALS", 0) == "MKTIIALS"


# ── Config validators ────────────────────────────────────────────────────────


def test_config_rejects_strong_rank_above_weak_rank():
    """A strong threshold looser than the weak one is contradictory and rejected."""
    with pytest.raises(ValueError, match="must be <= weak_rank"):
        NetMHCpanPeptidesConfig(strong_rank=5.0, weak_rank=2.0)


def test_config_accepts_equal_thresholds():
    """Equal thresholds are allowed: every binder is then either strong or a non-binder."""
    assert NetMHCpanProteinConfig(strong_rank=2.0, weak_rank=2.0).strong_rank == 2.0


@pytest.mark.parametrize("length", [7, 15])
def test_protein_config_rejects_out_of_range_peptide_length(length):
    """k-mer lengths outside the class I 8-14 range are rejected."""
    with pytest.raises(ValueError, match="peptide lengths must be 8-14"):
        NetMHCpanProteinConfig(peptide_lengths=[length])


def test_protein_config_sorts_and_dedupes_peptide_lengths():
    """Peptide lengths are normalized to a sorted, de-duplicated list."""
    assert NetMHCpanProteinConfig(peptide_lengths=[11, 9, 9, 10]).peptide_lengths == [9, 10, 11]


def test_protein_config_rejects_empty_peptide_lengths():
    """An empty length list would enumerate no windows and is rejected."""
    with pytest.raises(ValueError, match="at least one length"):
        NetMHCpanProteinConfig(peptide_lengths=[])


# ── Derived binding propensity and classification ────────────────────────────


def test_binding_propensity_is_one_half_at_the_weak_threshold():
    """The logistic is centred on the weak threshold, so it is exactly 0.5 there."""
    assert binding_propensity(2.0, 2.0) == 0.5


def test_binding_propensity_strictly_decreasing_and_in_unit_range():
    """Propensity decreases strictly with %rank and stays within (0, 1]."""
    ranks = [0.0, 0.01, 0.1, 0.5, 2.0, 10.0, 100.0]
    values = [binding_propensity(rank, 2.0) for rank in ranks]
    assert all(0.0 < value <= 1.0 for value in values)
    assert all(later < earlier for earlier, later in itertools.pairwise(values))


def test_binding_propensity_tracks_the_weak_threshold():
    """Loosening the weak threshold raises the propensity assigned to a fixed %rank."""
    assert binding_propensity(2.0, 5.0) > binding_propensity(2.0, 2.0)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0.5, "strong"), (2.0, "weak"), (2.000001, "non-binder")],
)
def test_classify_threshold_boundaries_are_inclusive(rank, expected):
    """Both thresholds are inclusive: a rank equal to a cutoff takes the stricter class."""
    assert classify(rank, 0.5, 2.0) == expected


# ── Summary metrics ──────────────────────────────────────────────────────────


def test_summarize_counts_fraction_and_best_rank():
    """Counts, the strong fraction and best_rank are derived from the EL %ranks."""
    bindings = [_binding(0.1), _binding(0.3), _binding(1.0), _binding(50.0)]
    metrics = summarize(bindings, 2.0)
    assert metrics["strong_binder_count"] == 2
    assert metrics["weak_binder_count"] == 1
    assert metrics["strong_binder_fraction"] == 0.5
    assert metrics["best_rank"] == 0.1


def test_summarize_binding_propensity_is_the_mean_logistic():
    """The reported propensity is the mean of the per-pair logistic, re-derived here."""
    ranks = [0.1, 0.3, 1.0, 50.0]
    metrics = summarize([_binding(rank) for rank in ranks], 2.0)
    expected = sum(1.0 / (1.0 + (rank / 2.0) ** 2.0) for rank in ranks) / len(ranks)
    assert metrics["binding_propensity"] == pytest.approx(expected)


def test_summarize_empty_bindings_reports_no_binder():
    """With no predictions the counts are zero and best_rank falls back to the 100% floor."""
    metrics = summarize([], 2.0)
    assert metrics["strong_binder_count"] == 0
    assert metrics["strong_binder_fraction"] == 0.0
    assert metrics["best_rank"] == 100.0
    assert metrics["binding_propensity"] == 0.0


# ── stdout parsing (hand-written tables, not captured output) ────────────────


def test_parse_output_reads_el_columns_and_version():
    """EL columns are read by header name and the version banner is picked up."""
    version, records = _standalone_run.parse_output(_table(_EL_HEADER, _EL_ROW_SB, _EL_ROW_NO_LEVEL))
    assert version == "4.1b"
    assert len(records) == 2
    assert records[0]["Peptide"] == "AAAAAAAAV"
    assert records[0]["Score_EL"] == "0.9000000"
    assert records[0]["%Rank_EL"] == "0.100"
    assert "Score_BA" not in records[0]


def test_parse_output_folds_two_token_bind_level():
    """``<= SB`` spans two tokens and is folded into the final BindLevel column."""
    _, records = _standalone_run.parse_output(_table(_EL_HEADER, _EL_ROW_SB, _EL_ROW_NO_LEVEL))
    assert records[0]["BindLevel"] == "<= SB"


def test_parse_output_row_without_bind_level_yields_empty_level():
    """A non-binder row prints no BindLevel, which parses to an empty final column."""
    _, records = _standalone_run.parse_output(_table(_EL_HEADER, _EL_ROW_SB, _EL_ROW_NO_LEVEL))
    assert records[1]["BindLevel"] == ""
    assert records[1]["%Rank_EL"] == "40.000"


def test_parse_output_ba_columns_are_additive():
    """-BA adds Score_BA/%Rank_BA/Aff(nM) without shifting the eluted-ligand columns."""
    _, records = _standalone_run.parse_output(_table(_BA_HEADER, _BA_ROW_SB))
    record = records[0]
    assert record["%Rank_EL"] == "0.100"
    assert record["Score_BA"] == "0.700000"
    assert record["%Rank_BA"] == "0.300"
    assert record["Aff(nM)"] == "20.00"
    assert record["BindLevel"] == "<= SB"


def test_parse_output_handles_optional_exp_column():
    """An extra Exp column is keyed off the header, so %Rank_EL is still read correctly."""
    _, records = _standalone_run.parse_output(_table(_EXP_HEADER, _EXP_ROW_WB))
    record = records[0]
    assert record["%Rank_EL"] == "1.500"
    assert record["Exp"] == "0.800"
    assert record["BindLevel"] == "<= WB"


def test_parse_output_without_a_table_raises():
    """Output with no result table is a failure, not an empty result set."""
    with pytest.raises(ValueError, match="no result table"):
        _standalone_run.parse_output("# NetMHCpan version 4.1b\nError. Something went wrong.\n")


def test_binding_from_record_restates_allele_unstarred():
    """The starred allele the table prints is restated in the form a later call accepts."""
    _, records = _standalone_run.parse_output(_table(_EL_HEADER, _EL_ROW_SB))
    binding = binding_from_record(records[0], 0.5, 2.0)
    assert binding.allele == "HLA-A02:01"
    assert binding.binder_class == "strong"
    assert binding.score_ba is None


# ---------------------------------------------------------------------------
# Integration tests (require a local, license-gated netMHCpan install)
# ---------------------------------------------------------------------------


def _require_netmhcpan():
    """Skip the test when no netMHCpan install is resolvable on this host."""
    try:
        _standalone_run.resolve_install(None)
    except FileNotFoundError:
        pytest.skip("netMHCpan not installed; see the netmhcpan toolkit SETUP.md to provision it")


def _error_text(exc):
    """Full text of a dispatch failure, including the standalone subprocess's stderr."""
    parts = (str(exc), getattr(exc, "stderr", None), getattr(exc, "stdout", None))
    return "\n".join(str(part) for part in parts if part)


@pytest.mark.integration
def test_netmhcpan_peptides_basic_execution():
    """A peptide run returns one index-aligned prediction per peptide per allele."""
    _require_netmhcpan()
    peptides = [_A0201_BINDER, _A0201_NON_BINDER]

    result = run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=peptides), NetMHCpanPeptidesConfig())

    assert result.success
    assert result.tool_id == "netmhcpan-peptides"
    assert result.netmhcpan_version
    assert [prediction.peptide for prediction in result.predictions] == peptides
    assert all(len(prediction.bindings) == 1 for prediction in result.predictions)
    assert all(prediction.bindings[0].allele == "HLA-A02:01" for prediction in result.predictions)


@pytest.mark.integration
def test_netmhcpan_peptides_known_binder_ranks_below_known_non_binder():
    """A canonical A*02:01 epitope clears the strong cutoff; a B7-restricted one does not."""
    _require_netmhcpan()
    config = NetMHCpanPeptidesConfig()

    result = run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=[_A0201_BINDER, _A0201_NON_BINDER]), config)
    binder, non_binder = (prediction.bindings[0] for prediction in result.predictions)

    assert binder.rank_el <= config.strong_rank
    assert binder.binder_class == "strong"
    assert non_binder.rank_el > config.strong_rank
    assert non_binder.binder_class != "strong"
    assert binder.rank_el < non_binder.rank_el


@pytest.mark.integration
def test_netmhcpan_peptides_binding_affinity_flag_populates_ba_fields():
    """include_binding_affinity passes -BA, which adds the BA columns the default run lacks."""
    _require_netmhcpan()
    inputs = NetMHCpanPeptidesInput(peptides=[_A0201_BINDER])

    with_ba = run_netmhcpan_peptides(inputs, NetMHCpanPeptidesConfig(include_binding_affinity=True))
    el_only = run_netmhcpan_peptides(inputs, NetMHCpanPeptidesConfig())

    ba_binding = with_ba.predictions[0].bindings[0]
    assert ba_binding.score_ba is not None
    assert ba_binding.rank_ba is not None
    assert ba_binding.affinity_nm is not None and ba_binding.affinity_nm > 0.0

    el_binding = el_only.predictions[0].bindings[0]
    assert (el_binding.score_ba, el_binding.rank_ba, el_binding.affinity_nm) == (None, None, None)
    # -BA is additive, so the eluted-ligand prediction must be unchanged by the flag.
    assert ba_binding.rank_el == el_binding.rank_el


@pytest.mark.integration
def test_netmhcpan_protein_windowing_matches_explicit_peptide_list():
    """Scanning a protein equals scoring the same k-mers passed in explicitly."""
    _require_netmhcpan()
    kmers = [_PROTEIN[start : start + _WINDOW_LENGTH] for start in range(len(_PROTEIN) - _WINDOW_LENGTH + 1)]

    scan = run_netmhcpan_protein(
        NetMHCpanProteinInput(sequences=[_PROTEIN]),
        NetMHCpanProteinConfig(peptide_lengths=[_WINDOW_LENGTH]),
    )
    explicit = run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=kmers), NetMHCpanPeptidesConfig())

    by_position = {binding.position: binding for binding in scan.results[0].bindings}
    assert [by_position[index + 1].peptide for index in range(len(kmers))] == kmers
    assert [by_position[index + 1].rank_el for index in range(len(kmers))] == [
        prediction.bindings[0].rank_el for prediction in explicit.predictions
    ]


@pytest.mark.integration
def test_netmhcpan_protein_window_count_and_one_indexed_positions():
    """A single-length scan yields L-k+1 windows, positioned 1-indexed from the sequence start."""
    _require_netmhcpan()

    result = run_netmhcpan_protein(
        NetMHCpanProteinInput(sequences=[_PROTEIN]),
        NetMHCpanProteinConfig(peptide_lengths=[_WINDOW_LENGTH]),
    )
    scan = result.results[0]
    positions = [binding.position for binding in scan.bindings]

    assert scan.sequence_length == len(_PROTEIN)
    assert scan.num_windows == len(_PROTEIN) - _WINDOW_LENGTH + 1
    assert min(positions) == 1
    assert max(positions) == len(_PROTEIN) - _WINDOW_LENGTH + 1
    first = next(binding for binding in scan.bindings if binding.position == 1)
    assert first.peptide == _PROTEIN[:_WINDOW_LENGTH]


@pytest.mark.integration
def test_netmhcpan_peptides_multi_allele_panel_in_configured_order():
    """A panel is scored in one call, one binding per allele per peptide, in configured order."""
    _require_netmhcpan()
    config = NetMHCpanPeptidesConfig(alleles=["HLA-B07:02", "HLA-A*02:01", "HLA-C07:01"])

    result = run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=[_A0201_BINDER, _A0201_NON_BINDER]), config)

    assert config.alleles == ["HLA-B07:02", "HLA-A02:01", "HLA-C07:01"]
    for prediction in result.predictions:
        assert [binding.allele for binding in prediction.bindings] == config.alleles
        assert all(binding.peptide == prediction.peptide for binding in prediction.bindings)


@pytest.mark.integration
def test_netmhcpan_peptides_duplicate_peptides_are_index_aligned():
    """A repeated peptide gets its own prediction at each position, with identical values."""
    _require_netmhcpan()

    result = run_netmhcpan_peptides(
        NetMHCpanPeptidesInput(peptides=[_A0201_BINDER, _A0201_NON_BINDER, _A0201_BINDER]),
        NetMHCpanPeptidesConfig(),
    )

    assert len(result.predictions) == 3
    first, _, repeat = result.predictions
    assert repeat.peptide == first.peptide
    assert repeat.bindings[0].rank_el == first.bindings[0].rank_el
    assert repeat.binding_propensity == first.binding_propensity


@pytest.mark.integration
def test_netmhcpan_peptides_unknown_allele_error_names_the_allele():
    """A nonsense allele is reported by name, not as an opaque binary or file failure."""
    _require_netmhcpan()
    config = NetMHCpanPeptidesConfig(alleles=["HLA-Z99:99"])

    with pytest.raises((ValueError, subprocess.CalledProcessError)) as excinfo:
        run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=[_A0201_BINDER]), config)

    # The one-shot dispatch path re-raises as CalledProcessError, carrying the standalone's
    # own message on .stderr, so the actionable text is asserted rather than the type.
    message = _error_text(excinfo.value)
    assert "HLA-Z99:99" in message
    assert "Unknown allele" in message


@pytest.mark.integration
def test_netmhcpan_misconfigured_install_error_names_the_config_field(tmp_path):
    """An explicit netmhcpan_dir that holds no install fails naming the field, not the binary."""
    _require_netmhcpan()
    config = NetMHCpanPeptidesConfig(netmhcpan_dir=str(tmp_path / "no-such-install"))

    with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)) as excinfo:
        run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=[_A0201_BINDER]), config)

    message = _error_text(excinfo.value)
    assert "netmhcpan_dir" in message
    assert "no-such-install" in message


@pytest.mark.integration
def test_netmhcpan_peptides_export_csv_and_json(tmp_path):
    """Export writes one CSV row per peptide-allele pair plus a header, and a JSON dump."""
    _require_netmhcpan()
    import csv

    peptides = [_A0201_BINDER, _A0201_NON_BINDER]
    config = NetMHCpanPeptidesConfig(alleles=["HLA-A02:01", "HLA-B07:02"])
    result = run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=peptides), config)

    result.export(name="netmhcpan", export_path=tmp_path, file_format="csv")
    result.export(name="netmhcpan", export_path=tmp_path, file_format="json")

    csv_path = tmp_path / "netmhcpan.csv"
    json_path = tmp_path / "netmhcpan.json"
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == len(peptides) * len(config.alleles) + 1
    assert rows[0][:2] == ["peptide", "allele"]


# ── Upstream reference outputs ───────────────────────────────────────────────
#
# The netMHCpan archive ships a test/ directory of reference outputs that DTU produced
# (the generating commands are listed in test/comm). They are an oracle the wrapper did
# not generate, so comparing against them exercises the parser, the allele round-trip,
# the FASTA windowing and the synthetic-identity mapping all at once. The files are read
# from the user's own licensed install at run time and never committed here, so no
# netMHCpan output is redistributed.


def _upstream_reference(name: str) -> list[dict[str, str]]:
    """Parse one reference output shipped in the install's test/ directory."""
    binary, _ = _standalone_run.resolve_install(None)
    reference = Path(binary).parent.parent.parent / "test" / name
    if not reference.is_file():
        pytest.skip(f"upstream reference {name} is not present in this install")
    return _standalone_run.parse_output(reference.read_text())[1]


# Upstream's BindLevel annotation, in our vocabulary.
_UPSTREAM_BIND_LEVELS = {"<= SB": "strong", "<= WB": "weak", "": "non-binder"}


@pytest.mark.integration
def test_matches_upstream_reference_el_peptides():
    """Peptide EL scores, ranks and binder classes match DTU's shipped test.pep.out."""
    _require_netmhcpan()
    reference = _upstream_reference("test.pep.out")
    peptides = [row["Peptide"] for row in reference]

    result = run_netmhcpan_peptides(
        NetMHCpanPeptidesInput(peptides=peptides),
        NetMHCpanPeptidesConfig(alleles=["HLA-A02:01"]),
    )

    ours = {pred.peptide: pred.bindings[0] for pred in result.predictions}
    assert len(ours) == len(reference)
    for row in reference:
        binding = ours[row["Peptide"]]
        assert binding.score_el == float(row["Score_EL"])
        assert binding.rank_el == float(row["%Rank_EL"])
        # binder_class is derived from rank_el; this asserts it agrees with the
        # BindLevel column netMHCpan printed.
        assert binding.binder_class == _UPSTREAM_BIND_LEVELS[row["BindLevel"]]


@pytest.mark.integration
def test_matches_upstream_reference_binding_affinity():
    """BA score, rank and nM affinity match DTU's shipped test.pep_wBA.out."""
    _require_netmhcpan()
    reference = _upstream_reference("test.pep_wBA.out")
    peptides = [row["Peptide"] for row in reference]

    result = run_netmhcpan_peptides(
        NetMHCpanPeptidesInput(peptides=peptides),
        NetMHCpanPeptidesConfig(alleles=["HLA-A02:01"], include_binding_affinity=True),
    )

    ours = {pred.peptide: pred.bindings[0] for pred in result.predictions}
    for row in reference:
        binding = ours[row["Peptide"]]
        assert binding.score_ba == float(row["Score_BA"])
        assert binding.rank_ba == float(row["%Rank_BA"])
        assert binding.affinity_nm == float(row["Aff(nM)"])


@pytest.mark.integration
def test_matches_upstream_reference_fasta_windowing():
    """Every window of DTU's shipped test.fsa.out is reproduced at the same position."""
    _require_netmhcpan()
    reference = _upstream_reference("test.fsa.out")
    binary, _ = _standalone_run.resolve_install(None)
    fasta = Path(binary).parent.parent.parent / "test" / "test.fsa"

    sequences: list[str] = []
    for raw_line in fasta.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            sequences.append("")
        elif line:
            sequences[-1] += line

    # Upstream identities are the FASTA headers with punctuation collapsed and truncated
    # to 15 characters, which is why the wrapper writes synthetic ids instead; records
    # come back in input order, so map identities by first appearance.
    identity_index: dict[str, int] = {}
    for row in reference:
        identity_index.setdefault(row["Identity"], len(identity_index))
    expected = {
        (identity_index[row["Identity"]], int(row["Pos"]), row["Peptide"]): (
            float(row["Score_EL"]),
            float(row["%Rank_EL"]),
        )
        for row in reference
    }

    # test.fsa.out was generated with netMHCpan's defaults, which the config mirrors.
    result = run_netmhcpan_protein(
        NetMHCpanProteinInput(sequences=sequences),
        NetMHCpanProteinConfig(alleles=["HLA-A02:01"]),
    )
    ours = {
        (index, binding.position, binding.peptide): (binding.score_el, binding.rank_el)
        for index, scan in enumerate(result.results)
        for binding in scan.bindings
    }

    assert ours == expected


# ── Benchmarks ───────────────────────────────────────────────────────────────


@pytest.mark.benchmark("netmhcpan-peptides")
@pytest.mark.slow
@pytest.mark.integration
def test_netmhcpan_peptides_benchmark(request: pytest.FixtureRequest):
    """Benchmark a peptide library against a three-allele panel (cold + warm)."""
    _require_netmhcpan()
    peptides = random_protein_sequences(512, _WINDOW_LENGTH, seed=0)
    config = NetMHCpanPeptidesConfig(alleles=["HLA-A02:01", "HLA-B07:02", "HLA-C07:01"])

    result = benchmark_twice(
        request,
        "netmhcpan",
        lambda: run_netmhcpan_peptides(NetMHCpanPeptidesInput(peptides=peptides), config),
    )

    assert result.tool_id == "netmhcpan-peptides"
    assert len(result.predictions) == len(peptides)
    assert all(len(prediction.bindings) == len(config.alleles) for prediction in result.predictions)


@pytest.mark.benchmark("netmhcpan-protein")
@pytest.mark.slow
@pytest.mark.integration
def test_netmhcpan_protein_benchmark(request: pytest.FixtureRequest):
    """Benchmark a multi-length epitope scan over a batch of proteins (cold + warm)."""
    _require_netmhcpan()
    sequences = random_protein_sequences(16, 200, seed=1)
    config = NetMHCpanProteinConfig(alleles=["HLA-A02:01", "HLA-B07:02"], peptide_lengths=[9, 10])

    result = benchmark_twice(
        request,
        "netmhcpan",
        lambda: run_netmhcpan_protein(NetMHCpanProteinInput(sequences=sequences), config),
    )

    assert result.tool_id == "netmhcpan-protein"
    assert len(result.results) == len(sequences)
    expected_windows = sum(200 - length + 1 for length in config.peptide_lengths)
    assert result.results[0].num_windows == expected_windows
