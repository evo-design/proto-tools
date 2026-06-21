"""tests/structure_prediction_tests/test_x3dna_fiber.py.

Tests for the X3DNA Fiber idealized nucleic-acid duplex builder.

Validation and helper coverage runs without any external binary; the execution
tests are marked ``integration`` and skip cleanly when X3DNA is not resolvable.
"""

import os

import pytest

from proto_tools.tools import (
    X3DNAFiberConfig,
    X3DNAFiberInput,
    run_x3dna_fiber,
)
from proto_tools.tools.structure_prediction.x3dna.x3dna_fiber import (
    _FORM_FLAGS,
    _normalize_bases,
)
from tests.conftest import benchmark_twice

# ── Constants ────────────────────────────────────────────────────────────────
_DUPLEX_SEQ = "GGGCAAAATGCACTGCACTTTGGG"


# ── Input validation (custom logic, not Pydantic boilerplate) ────────────────


def test_input_single_string_normalizes_to_list():
    """A single sequence string is normalized to a one-element list."""
    assert X3DNAFiberInput(sequences="ACGT").sequences == ["ACGT"]


def test_input_empty_list_raises():
    """An empty list of sequences is rejected."""
    with pytest.raises(ValueError, match="at least one sequence"):
        X3DNAFiberInput(sequences=[])


def test_input_empty_sequence_string_raises():
    """An empty string inside the list is rejected."""
    with pytest.raises(ValueError, match="empty"):
        X3DNAFiberInput(sequences=[""])


def test_input_invalid_bases_raise():
    """Bases outside A/C/G/T/U are rejected with the offending base reported."""
    with pytest.raises(ValueError, match="invalid base"):
        X3DNAFiberInput(sequences=["ACGTX"])


def test_input_accepts_all_valid_bases_any_case():
    """A/C/G/T/U in mixed case pass validation and are preserved as-given."""
    inputs = X3DNAFiberInput(sequences=["acgtu", "ACGTU"])
    assert inputs.sequences == ["acgtu", "ACGTU"]


# ── _normalize_bases helper ──────────────────────────────────────────────────


def test_normalize_bases_rna_converts_t_to_u():
    """RNA form upper-cases and converts T->U (and leaves U)."""
    assert _normalize_bases("acgtu", "RNA") == "ACGUU"


@pytest.mark.parametrize("form", ["A-DNA", "B-DNA", "Z-DNA"])
def test_normalize_bases_dna_converts_u_to_t(form):
    """DNA forms upper-case and convert U->T (and leave T)."""
    assert _normalize_bases("acgtu", form) == "ACGTT"


# ── _FORM_FLAGS mapping ──────────────────────────────────────────────────────


def test_form_flags_mapping():
    """Each canonical form maps to its expected fiber CLI flag."""
    assert _FORM_FLAGS == {"A-DNA": "-a", "B-DNA": "-b", "Z-DNA": "-z", "RNA": "-rna"}


# ── Config cloud support ─────────────────────────────────────────────────────


def test_config_cloud_unsupported():
    """X3DNA is a local-only binary; cloud execution must be reported unsupported."""
    reason = X3DNAFiberConfig().cloud_unsupported_reason()
    assert reason is not None
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# Integration tests (require a local X3DNA install)
# ---------------------------------------------------------------------------


def _require_x3dna():
    """Skip the test when no X3DNA install is resolvable on this host."""
    if not os.environ.get("X3DNA"):
        pytest.skip("X3DNA not installed (set $X3DNA to a v2.4 install root)")


@pytest.mark.integration
def test_x3dna_fiber_builds_b_dna_duplex():
    """B-DNA build yields one duplex structure with two chains of 2x the sequence."""
    _require_x3dna()

    output = run_x3dna_fiber(
        X3DNAFiberInput(sequences=[_DUPLEX_SEQ]),
        X3DNAFiberConfig(form="B-DNA"),
    )

    assert output.tool_id == "x3dna-fiber"
    assert len(output.structures) == 1

    structure = output.structures[0]
    assert structure.num_chains == 2  # duplex: sense + generated complement
    assert structure.num_residues == 2 * len(_DUPLEX_SEQ)


@pytest.mark.integration
def test_x3dna_fiber_single_stranded_yields_one_chain():
    """single_stranded=True emits only the sense strand (one chain, N residues)."""
    _require_x3dna()

    output = run_x3dna_fiber(
        X3DNAFiberInput(sequences=[_DUPLEX_SEQ]),
        X3DNAFiberConfig(form="B-DNA", single_stranded=True),
    )

    assert len(output.structures) == 1
    structure = output.structures[0]
    assert structure.num_chains == 1
    assert structure.num_residues == len(_DUPLEX_SEQ)


@pytest.mark.integration
def test_x3dna_fiber_rna_form_builds_duplex():
    """RNA form builds an A-form RNA duplex (two chains, 2x the sequence)."""
    _require_x3dna()

    output = run_x3dna_fiber(
        X3DNAFiberInput(sequences=[_DUPLEX_SEQ]),
        X3DNAFiberConfig(form="RNA"),
    )

    assert len(output.structures) == 1
    structure = output.structures[0]
    assert structure.num_chains == 2
    assert structure.num_residues == 2 * len(_DUPLEX_SEQ)


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("x3dna-fiber")
@pytest.mark.slow
@pytest.mark.integration
def test_x3dna_fiber_benchmark(request: pytest.FixtureRequest):
    """Benchmark x3dna-fiber building a batch of B-DNA duplexes (cold + warm)."""
    _require_x3dna()
    bases = "ACGT"
    sequences = ["".join(bases[(i * 7 + j) % 4] for j in range(30)) for i in range(64)]

    result = benchmark_twice(
        request,
        "x3dna",
        lambda: run_x3dna_fiber(X3DNAFiberInput(sequences=sequences), X3DNAFiberConfig(form="B-DNA")),
    )

    assert result.tool_id == "x3dna-fiber"
    assert len(result.structures) == len(sequences)
    assert result.structures[0].num_chains == 2
