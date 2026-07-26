"""Tests for AlphaFold3 structural-template support.

The PR adds an optional ``templates`` field to ``AlphaFold3Input`` so a caller can supply a
per-chain structural template (AF3's native ``{mmcif, queryIndices, templateIndices}``) — the
motivating use is pinning a chopped domain to its already-good monomer conformation instead of
letting AF3 re-predict it from scratch.

The make-or-break unknown was whether AF3 consumes a populated ``templates`` array under
``--norun_data_pipeline`` (the flag the wrapper always passes).
``test_template_is_honored_under_norun_data_pipeline`` is that gate — folding a chain with its own
structure as template and confirming, via SUPERPOSED Cα RMSD against a run-to-run baseline, that
the template steers the fold. **Verified on hardware: 1.48 Å templated vs 16.60 Å baseline** — AF3
honors the template. (Getting here also required the mmCIF the template ships to be single-chain,
have integer label_seq_id/entity_poly_seq, and carry a release date; ``make_af3_template`` handles
all three.)

The wiring tests run anywhere (no GPU/AF3); the gate test skips unless AF3 is runnable.
"""

from __future__ import annotations

import os

import pytest

from proto_tools.entities.complex import Chain, Complex
from proto_tools.tools.structure_prediction.alphafold3.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Input,
    _assign_templates_to_input_json,
    _create_input_json_from_complex,
    run_alphafold3,
)


def _af3_available() -> bool:
    """AF3 is runnable if weights + (sif or env install) are configured."""
    return bool(os.environ.get("PROTO_ALPHAFOLD3_WEIGHTS_DIR")) and (
        bool(os.environ.get("PROTO_ALPHAFOLD3_SIF_PATH")) or bool(os.environ.get("PROTO_ALPHAFOLD3_STANDALONE_DIR"))
    )


# --------------------------------------------------------------------------------------------
# Wiring (no GPU): the template dict must land in the right protein chain and nowhere else.
# --------------------------------------------------------------------------------------------
def test_templates_field_defaults_to_none():
    cx = Complex(chains=[Chain(id="A", sequence="MKTLLILAVVAAALA", entity_type="protein")])
    assert AlphaFold3Input(complexes=[cx]).templates is None


def test_assign_templates_routes_to_correct_chain():
    cx = Complex(
        chains=[
            Chain(id="A", sequence="MKTLLILAVVAAALA", entity_type="protein"),
            Chain(id="B", sequence="GSSGSSGQWERTYIP", entity_type="protein"),
        ]
    )
    j = _create_input_json_from_complex(cx, "t", [0])
    # every protein chain starts with the required empty templates list
    assert all(s["protein"]["templates"] == [] for s in j["sequences"] if "protein" in s)

    tmpl = {0: [{"mmcif": "data_x", "queryIndices": [0, 1, 2], "templateIndices": [0, 1, 2]}]}
    j = _assign_templates_to_input_json(j, tmpl, cx)
    assert j["sequences"][0]["protein"]["templates"][0]["queryIndices"] == [0, 1, 2]
    assert j["sequences"][1]["protein"]["templates"] == []  # chain B untouched


def test_assign_templates_ignores_missing_chain_index():
    cx = Complex(chains=[Chain(id="A", sequence="MKTLLILAVVAAALA", entity_type="protein")])
    j = _create_input_json_from_complex(cx, "t", [0])
    # chain index 5 does not exist; must be a no-op, not an error, and template nothing
    j = _assign_templates_to_input_json(j, {5: [{"mmcif": "x", "queryIndices": [], "templateIndices": []}]}, cx)
    assert j["sequences"][0]["protein"]["templates"] == []


# --------------------------------------------------------------------------------------------
# GATE (needs AF3): is a populated template actually consumed under --norun_data_pipeline?
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _af3_available(), reason="AF3 weights/install not configured")
def test_template_is_honored_under_norun_data_pipeline(tmp_path):
    """Fold with a template and confirm it STEERS the output, under --norun_data_pipeline.

    The gate metric is SUPERPOSED Ca RMSD (raw RMSD is frame-dependent and meaningless across two
    AF3 outputs). Discriminating design: fold the sequence untemplated (ref), then fold it again
    (a) with ref as template and (b) untemplated with a different seed as the run-to-run baseline.
    If the template is honored, the templated fold sits far closer to the template than the
    baseline does. Verified on hardware: 1.48 A templated vs 16.60 A baseline.
    """
    import gemmi
    from proto_tools.tools.structure_prediction.alphafold3.alphafold3 import make_af3_template

    seq = (
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
        "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG"
    )
    cx = Complex(chains=[Chain(id="A", sequence=seq, entity_type="protein")])

    def fold(seed, templates=None):
        cfg = AlphaFold3Config(name=f"g{seed}", use_msa=False, num_diffusion_samples=1, seed=seed, verbose=0)
        return run_alphafold3(AlphaFold3Input(complexes=[cx], templates=templates), cfg).structures[0]

    def ca(s):
        m = gemmi.read_structure_string(s.structure_cif)[0]
        return [r.sole_atom("CA").pos for ch in m for r in ch if r.find_atom("CA", "*")]

    def surmsd(a, b):
        pa, pb = ca(a), ca(b); n = min(len(pa), len(pb))
        return gemmi.superpose_positions(pa[:n], pb[:n]).rmsd

    ref = fold(1)
    ref_pdb = tmp_path / "ref.pdb"; ref.write_pdb(ref_pdb)
    tmpl = {0: [make_af3_template(str(ref_pdb), "A", list(range(len(seq))), list(range(len(seq))))]}
    templated = fold(2, templates=[tmpl])
    baseline = fold(2)

    r_tmpl = surmsd(ref, templated)
    r_base = surmsd(ref, baseline)
    assert r_tmpl < 3.0 and r_tmpl < r_base - 1.0, (
        f"template did not steer the fold: templated {r_tmpl:.1f} A vs baseline {r_base:.1f} A"
    )
