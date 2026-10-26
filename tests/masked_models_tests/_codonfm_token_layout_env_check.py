"""In-env check that the CodonFM gradient builds the token layout the model expects.

``compute_gradient`` cannot hand Encodon a distribution, so it assembles the input itself:
``[CLS] + codons + [SEP]`` embeddings padded with ``[PAD]`` to a multiple of eight, a parallel
``token_ids`` row, and masked-position indices offset by the one leading special token. A drift
there — a missing ``CLS``, a ``SEP`` in the wrong place, a padding rule that stops matching the
scoring paths — would leave the gradient finite and self-consistent while scoring a sequence the
model never sees, so no finite-difference or shape test would notice.

``_codonfm_vocabulary_env_check.py`` already pins the 64 codon *columns*; this pins the special
tokens around them. ``Tokenizer()`` takes no checkpoint, so this needs no weights and no GPU — it
runs inside the codonfm standalone env only because the package lives there.

Run by ``test_codonfm_gradient_token_layout_matches_the_tokenizer`` in ``test_codonfm.py``. Plain
asserts; a non-zero exit surfaces as ``RuntimeError``. The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

from __future__ import annotations

from src.inference.encodon import Tokenizer  # type: ignore[import-not-found]

CDS = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"  # 30 nt, 10 codons

tokenizer = Tokenizer()
codons = [CDS[index : index + 3] for index in range(0, len(CDS), 3)]

cls_id = int(tokenizer.cls_token_id)
sep_id = int(tokenizer.sep_token_id)
pad_id = int(tokenizer.pad_token_id)
mask_id = int(tokenizer.mask_token_id)

# What the worker assembles before padding: one leading CLS, the codon ids gathered through the
# tokenizer's own encoder, one trailing SEP.
built = [cls_id] + [int(tokenizer.encoder[codon]) for codon in codons] + [sep_id]

assert len(built) == len(codons) + 2, f"expected one leading and one trailing special, got {len(built)}"
assert built[0] == cls_id, "first token must be CLS"
assert built[-1] == sep_id, "last token must be SEP"

# The four special ids must be distinct from each other and from every codon column, otherwise
# masking or padding would silently overwrite a real codon.
codon_ids = {int(value) for value in tokenizer.encoder.values()}
specials = {"cls": cls_id, "sep": sep_id, "pad": pad_id, "mask": mask_id}
assert len(set(specials.values())) == len(specials), f"special ids collide: {specials}"
for name, token_id in specials.items():
    assert token_id not in {int(tokenizer.encoder[codon]) for codon in codons}, (
        f"{name} id {token_id} collides with a codon in the sequence"
    )

# The gradient pads to the same multiple-of-eight context the scoring paths use:
# ``min(2048, ceil((codons + 2) / 8) * 8)``.
expected_context = min(2048, ((len(codons) + 2 + 7) // 8) * 8)
assert expected_context % 8 == 0, "context length must be a multiple of eight"
assert expected_context >= len(built), f"context {expected_context} cannot hold {len(built)} tokens"

print("CODONFM_TOKEN_LAYOUT_OK")
