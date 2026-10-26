"""In-env check that the CodonFM gradient wraps codons in the special tokens Encodon expects.

``compute_gradient`` assembles the model input by hand rather than tokenizing, so the special
tokens around the codon row are hand-placed. A drift there — a missing ``CLS``, ``SEP`` in the
wrong slot, a padding rule that stops matching the scoring paths — would leave the gradient
finite, correctly shaped and self-consistent under finite difference, while scoring a sequence
the model never sees.

This calls the worker's **own** ``build_model_input_row`` rather than restating the layout, and
compares both the ids and the attention mask against ``process_item`` — the preprocessing every
Encodon scoring path runs a CDS through. Because both sides go through shipped code, changing the
worker changes this test's input; a reimplementation here would only have verified the docstring.
The raw tokenizer adds no special tokens at all, so the wrapping exists only in those two places
and they are the pair that must agree.

Unlike ESM2 and AbLang there is no whole-sequence scoring tool to cross-check against:
``codonfm-score`` scores mutations, and ``codonfm-fitness`` reports a different quantity than the
gradient's masked PLL (measured: -4.354665 vs -4.063083 on this CDS). So the layout is pinned
directly. ``Tokenizer()`` takes no checkpoint, so this needs no weights and no GPU.

Run by ``test_codonfm_gradient_token_layout_matches_the_scoring_path`` in ``test_codonfm.py``. Plain
asserts; a non-zero exit surfaces as ``RuntimeError``. The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

from __future__ import annotations

import inference  # standalone module; on PYTHONPATH inside the codonfm env
from src.data.preprocess.codon_sequence import process_item  # type: ignore[import-not-found]
from src.inference.encodon import Tokenizer  # type: ignore[import-not-found]
from standalone_helpers.vocabulary import DNA_CODONS

CDS = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"  # 30 nt, 10 codons

tokenizer = Tokenizer()
codons = [CDS[index : index + 3] for index in range(0, len(CDS), 3)]
codon_ids = [int(tokenizer.encoder[codon]) for codon in codons]

# The shipped assembly the gradient feeds the model, called directly.
built_ids, built_mask = inference.build_model_input_row(tokenizer, codon_ids)

# What the scoring paths feed the model for the same CDS. ``process_item`` is the authority
# here, not ``tokenizer.encode`` — the raw tokenizer adds no special tokens at all, so the
# wrapping only exists in these two places and must agree between them.
reference = process_item(CDS, len(built_ids), tokenizer)
reference_ids = [int(value) for value in reference["input_ids"]]
reference_mask = [int(value) for value in reference["attention_mask"]]

assert built_ids == reference_ids, f"token layout drifted:\n  gradient {built_ids}\n  scoring  {reference_ids}"
assert built_mask == reference_mask, f"attention mask drifted:\n  gradient {built_mask}\n  scoring  {reference_mask}"

cls_id, sep_id = inference.special_token_layout(tokenizer)
assert built_ids[0] == cls_id, "first token must be CLS"
assert built_ids[len(codons) + 1] == sep_id, "SEP must follow the last codon"
assert built_ids[len(codons) + 2 :] == [int(tokenizer.pad_token_id)] * (len(built_ids) - len(codons) - 2)
assert sum(built_mask) == len(codons) + 2, "only CLS, the codons and SEP may be attended to"

# The four special ids must be mutually distinct and must not collide with one of the 64 codon
# columns the gradient spans, or masking and padding would silently overwrite a real codon.
specials = {
    "cls": cls_id,
    "sep": sep_id,
    "pad": int(tokenizer.pad_token_id),
    "mask": int(tokenizer.mask_token_id),
}
assert len(set(specials.values())) == len(specials), f"special ids collide: {specials}"
all_codon_ids = {int(tokenizer.encoder[codon]) for codon in DNA_CODONS}
for name, token_id in specials.items():
    assert token_id not in all_codon_ids, f"{name} id {token_id} collides with a codon column"

assert len(built_ids) % 8 == 0, f"context length {len(built_ids)} is not a multiple of eight"

print("CODONFM_TOKEN_LAYOUT_OK")
