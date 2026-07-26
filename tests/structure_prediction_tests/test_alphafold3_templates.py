"""Tests for AlphaFold3 structural-template support.

The PR adds an optional ``templates`` field to ``AlphaFold3Input`` so a caller can supply a
per-chain structural template (AF3's native ``{mmcif, queryIndices, templateIndices}``) — the
motivating use is pinning a chopped domain to its already-good monomer conformation instead of
letting AF3 re-predict it from scratch.

There is one make-or-break unknown that gates the whole feature: **does AF3 actually consume a
populated ``templates`` array when we run it with ``--norun_data_pipeline``?** The wrapper always
passes that flag, and the only established fact so far is that ``templates`` must be *present*
(even ``[]``). ``test_template_is_honored_under_norun_data_pipeline`` is that gate: it folds one
chain with its OWN native structure as the template and asserts the prediction snaps to it
(low Cα RMSD). If AF3 ignores templates in this mode the gate fails and the feature is blocked —
do not build on it until this passes.

The wiring test runs anywhere (no GPU/AF3); the gate test skips unless AF3 is runnable.
"""

from __future__ import annotations

import os

import numpy as np
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
    """Fold a chain with its own native structure as template; output must match the template.

    This is the feature gate. AF3 diffusion is stochastic, so an untemplated fold of a short
    generic sequence would NOT reliably reproduce a specific structure; if the templated fold
    lands within a few Å Cα RMSD of the template, the template was used. A high RMSD means AF3
    ignored it under --norun_data_pipeline and the feature is blocked.
    """
    from proto_tools.entities.structures.structure import Structure

    seq = (
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
        "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG"
    )
    cx = Complex(chains=[Chain(id="A", sequence=seq, entity_type="protein")])
    cfg = AlphaFold3Config(name="tmpl_gate", use_msa=False, num_diffusion_samples=1, verbose=1)

    # 1) untemplated reference fold -> use it as the template for a second fold
    ref = run_alphafold3(AlphaFold3Input(complexes=[cx]), cfg).structures[0]
    ref_cif = tmp_path / "ref.cif"
    ref.write_cif(ref_cif)

    tmpl = {
        0: [{
            "mmcif": ref_cif.read_text(),
            "queryIndices": list(range(len(seq))),
            "templateIndices": list(range(len(seq))),
        }]
    }
    templated = run_alphafold3(AlphaFold3Input(complexes=[cx], templates=[tmpl]), cfg).structures[0]

    rmsd = _ca_rmsd(ref, templated)
    assert rmsd < 3.0, f"templated fold did not follow the template (Cα RMSD {rmsd:.1f} Å) — AF3 likely ignored templates under --norun_data_pipeline"


def _ca_rmsd(a, b) -> float:
    import gemmi

    def cas(struct):
        m = gemmi.read_structure_string(struct.structure_cif if hasattr(struct, "structure_cif") else str(struct))[0]
        return np.array([r.sole_atom("CA").pos.tolist() for ch in m for r in ch if r.find_atom("CA", "*")])

    x, y = cas(a), cas(b)
    n = min(len(x), len(y))
    return float(np.sqrt(((x[:n] - y[:n]) ** 2).sum(axis=1).mean()))
