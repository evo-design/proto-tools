"""In-env check that the ESM C gradient builds the token layout the model expects.

``compute_gradient`` cannot hand ESM C a distribution, so it assembles the input itself:
``[cls] + residues + [eos]`` embeddings, a parallel ``token_ids`` row, and masked-position
indices offset by the one leading special token. If that layout drifted from what the
tokenizer produces — a missing ``cls``, a trailing ``eos`` in the wrong place, a different
special-token id — the gradient would still be finite and self-consistent (a finite-difference
check would pass), while scoring a sequence the model never actually sees.

``EsmcTokenizer()`` needs no checkpoint and no GPU; this runs inside the shared ESM env only
because the ``esm`` package lives there.

Run by ``test_esmc_gradient_token_layout_matches_the_tokenizer`` in ``test_esmc_gradient.py``.
Plain asserts; a non-zero exit surfaces as ``RuntimeError``. The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

from __future__ import annotations

from esm.models.esmc.tokenizer import EsmcTokenizer  # type: ignore[import-not-found]
from standalone_helpers.vocabulary import AMINO_ACIDS_LIST

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

tokenizer = EsmcTokenizer()
vocab = tokenizer.get_vocab()
amino_acid_token_ids = [vocab[residue] for residue in AMINO_ACIDS_LIST]

# What the worker assembles: one leading cls, the residue ids gathered through proto's
# amino-acid column order, one trailing eos.
built = (
    [int(tokenizer.cls_token_id)]
    + [amino_acid_token_ids[AMINO_ACIDS_LIST.index(residue)] for residue in SEQUENCE]
    + [int(tokenizer.eos_token_id)]
)

# What the model's own tokenizer produces for the same sequence.
native = [int(token) for token in tokenizer(SEQUENCE, add_special_tokens=True)["input_ids"]]

assert built == native, f"token layout drifted:\n  built  {built}\n  native {native}"

# The worker scores positions 1..L, i.e. offset by exactly the one leading special token.
assert len(native) == len(SEQUENCE) + 2, f"expected one leading and one trailing special, got {len(native)}"
assert native[0] == int(tokenizer.cls_token_id), "first token must be cls"
assert native[-1] == int(tokenizer.eos_token_id), "last token must be eos"

# The mask token the worker substitutes must be a real, distinct vocabulary entry.
mask_id = int(tokenizer.mask_token_id)
assert mask_id not in amino_acid_token_ids, "mask id collides with an amino-acid column"
assert mask_id != int(tokenizer.cls_token_id) and mask_id != int(tokenizer.eos_token_id)

print("ESMC_TOKEN_LAYOUT_OK")
