"""tests/style_consistency_tests/test_vocabulary_consistency.py.

Pins proto's two canonical vocabulary sources to each other.

Column orders are declared twice by necessity: the parent spells them out in
``proto_tools/utils/sequence.py`` (the ``AminoAcid`` Literal has to be literal for static
typing and Pydantic), and the worker side keeps its own copy in
``standalone_helpers/vocabulary.py`` because standalone envs cannot import ``proto_tools``.
Neither can be derived from the other, so this module asserts they agree — and that no
toolkit has reintroduced a third copy.

CPU-only, no tool environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from standalone_helpers import vocabulary as worker_vocabulary

from proto_tools.tools.masked_models.codonfm.shared_data_models import (
    CODONFM_CODON_VOCAB,
    CodonSequenceLogits,
)
from proto_tools.tools.masked_models.mixed_data_models import MixedSequenceLogits
from proto_tools.utils.sequence import DNA_NUCLEOTIDES, PROTEIN_AMINO_ACIDS, RNA_NUCLEOTIDES
from proto_tools.utils.sequence_logits import NucleotideSequenceLogits, ProteinSequenceLogits

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "proto_tools" / "tools"
_WORKER_VOCABULARY = Path(worker_vocabulary.__file__).resolve()


# ── Parent and worker agree ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, parent_order, worker_order",
    [
        ("amino acids", list(PROTEIN_AMINO_ACIDS), worker_vocabulary.AMINO_ACIDS_LIST),
        ("DNA", list(DNA_NUCLEOTIDES), list(worker_vocabulary.DNA_NUCLEOTIDES)),
        ("RNA", list(RNA_NUCLEOTIDES), list(worker_vocabulary.RNA_NUCLEOTIDES)),
    ],
)
def test_parent_and_worker_vocabularies_match(name: str, parent_order: list[str], worker_order: list[str]) -> None:
    """The parent's declared order equals the worker's, symbol for symbol."""
    assert parent_order == worker_order, (
        f"The {name} column order differs between proto_tools/utils/sequence.py and "
        "standalone_helpers/vocabulary.py. Both are hand-declared because neither side can "
        "import the other; update whichever one is stale."
    )


# ── Relaxed-state types point at those orders ────────────────────────────────


@pytest.mark.parametrize(
    "matrix_type, expected",
    [
        (ProteinSequenceLogits, list(PROTEIN_AMINO_ACIDS)),
        (NucleotideSequenceLogits, list(DNA_NUCLEOTIDES)),
        (CodonSequenceLogits, CODONFM_CODON_VOCAB),
        (MixedSequenceLogits, worker_vocabulary.MIXED_VOCAB),
    ],
)
def test_sequence_logits_vocab_matches_its_source(matrix_type: type, expected: list[str]) -> None:
    """Each relaxed-state type's ``VOCAB`` is its declared order, not a separate copy."""
    assert list(matrix_type.VOCAB) == expected


def test_mixed_vocab_blocks_are_protein_then_dna() -> None:
    """The mixed order is protein in columns 0-19 and lowercase DNA in 20-23.

    ``MixedSequenceGradientInput.allowed_tokens`` masks by those column ranges, so the layout
    is load-bearing rather than cosmetic.
    """
    mixed = worker_vocabulary.MIXED_VOCAB
    assert mixed[:20] == list(PROTEIN_AMINO_ACIDS)
    assert mixed[20:] == [nucleotide.lower() for nucleotide in DNA_NUCLEOTIDES]


# ── No third copy ────────────────────────────────────────────────────────────

# Only proto's own interface order is guarded. A model's native ordering is declared in its
# toolkit's ``standalone/`` directory on purpose: those orders are fixed by separate upstreams
# and coincide only by accident, so one shared constant would couple the models wrongly.
_CANONICAL_ALPHABETS = {
    "ACDEFGHIKLMNPQRSTVWY": "standalone_helpers.vocabulary.AMINO_ACIDS_LIST",
}


# Vendored upstream code we don't fork, and mock toolkits whose alphabet is an arbitrary symbol
# set for a fake model rather than a biological vocabulary.
_EXEMPT_PARTS = ("__pycache__", "src", "testing")


def _python_sources() -> list[Path]:
    """Return every toolkit Python file whose alphabets proto actually owns."""
    return sorted(
        path
        for path in _TOOLS_DIR.rglob("*.py")
        if not any(part in _EXEMPT_PARTS for part in path.relative_to(_TOOLS_DIR).parts)
    )


def _string_literals(tree: ast.Module) -> list[str]:
    """Return every string constant in a parsed module."""
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


@pytest.mark.parametrize("source_path", _python_sources(), ids=lambda p: str(p.relative_to(_TOOLS_DIR)))
def test_toolkits_do_not_retype_a_canonical_alphabet(source_path: Path) -> None:
    """A toolkit references the canonical constant instead of spelling an alphabet out again.

    Retyping an alphabet is how a column order silently drifts: a transposed pair produces a
    valid-looking matrix whose gradient is attributed to the wrong symbols.
    """
    if source_path.resolve() == _WORKER_VOCABULARY:
        return
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    offenders = {literal: owner for literal, owner in _CANONICAL_ALPHABETS.items() if literal in _string_literals(tree)}

    assert not offenders, (
        f"{source_path.relative_to(_TOOLS_DIR)} spells out "
        + ", ".join(f"{literal!r} (owned by {owner})" for literal, owner in offenders.items())
        + ". Import the canonical constant instead; see notes/gradients.md."
    )
