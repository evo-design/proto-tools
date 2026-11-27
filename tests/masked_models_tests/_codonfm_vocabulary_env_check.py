"""In-env check that CodonFM's real tokenizer covers proto's codon column order.

The worker builds its column basis as ``[tokenizer.encoder[codon] for codon in DNA_CODONS]``, so
the gradient's columns are proto-ordered only if the real tokenizer maps every canonical codon by
name. ``Tokenizer()`` takes no checkpoint, so this needs no weights and no GPU — it runs inside
the codonfm standalone env only because the package lives there.

Run by ``test_codonfm_tokenizer_covers_the_canonical_codon_order`` in ``test_codonfm.py``. Plain
asserts; a non-zero exit surfaces as ``RuntimeError``. The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

from __future__ import annotations

import itertools

from src.inference.encodon import Tokenizer  # type: ignore[import-not-found]
from standalone_helpers.vocabulary import DNA_CODONS

CANONICAL = ["".join(codon) for codon in itertools.product("ACGT", repeat=3)]


def main() -> int:
    """Assert the tokenizer maps proto's 64 codons by name, and report the mapping."""
    assert DNA_CODONS == CANONICAL, f"shared order is not lexicographic: {DNA_CODONS[:4]}"

    tokenizer = Tokenizer()
    missing = [codon for codon in CANONICAL if codon not in tokenizer.encoder]
    assert not missing, f"tokenizer is missing codons {missing}"

    # Looked up by name, never by position, so the tokenizer's own iteration order cannot matter.
    token_ids = [tokenizer.encoder[codon] for codon in CANONICAL]
    assert len(set(token_ids)) == len(token_ids), "two canonical codons share one token id"
    assert all(isinstance(token_id, int) for token_id in token_ids), "token ids are not integers"

    # A stop codon is a real codon here: the gradient spans all 64, and only resampling excludes them.
    for stop in ("TAA", "TAG", "TGA"):
        assert stop in tokenizer.encoder, f"tokenizer is missing stop codon {stop}"

    print(f"PASS tokenizer covers {len(CANONICAL)} canonical codons; ids {token_ids[0]}..{token_ids[-1]}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
