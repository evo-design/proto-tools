"""tests/database_retrieval_tests/test_rfam.py.

Tests for the Rfam tools (rfam-family, rfam-regions).
"""

import json

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    RfamFamilyConfig,
    RfamFamilyInput,
    RfamRegionsConfig,
    RfamRegionsInput,
    run_rfam_family,
    run_rfam_regions,
)
from proto_tools.tools.database_retrieval.rfam import shared_data_models
from proto_tools.utils import base_config

BASE = "https://rfam.org/family"

REGIONS = (
    "# Rfam regions for family TwoAYGGAY (RF01731)\n"
    "# file built using Rfam version 15.1 (released 2026-01-16)\n"
    "# found 4 regions\n"
    "# columns: sequence accession, bits score, region start, region end, sequence description, species, NCBI tax ID\n"
    "AM181176.4\t66.1\t1720658\t1720784\tPseudomonas fluorescens SBW25 complete genome\tPseudomonas fluorescens SBW25\t216595\n"
    "AM181176.4\t65.4\t2297101\t2296975\tPseudomonas fluorescens SBW25 complete genome\tPseudomonas fluorescens SBW25\t216595\n"
    "CP000076.1\t60.2\t100\t220\tPseudomonas protegens Pf-5 complete genome\tPseudomonas protegens Pf-5\t220664\n"
    "CP000076.1\t58.0\t900\t1010\tPseudomonas protegens Pf-5 complete genome\tPseudomonas protegens Pf-5\t220664\n"
)

FAMILY = {
    "rfam": {
        "acc": "RF01731",
        "id": "TwoAYGGAY",
        "description": "TwoAYGGAY RNA",
        "comment": None,
        "curation": {
            "author": "Weinberg Z",
            "seed_source": "Published; PMID:20230605;",
            "num_seed": 210,
            "num_full": 1499,
            "num_species": 222,
            "type": "Cis-reg;",
            "structure_source": "Published; PMID:20230605;",
        },
        "cm": {"cutoffs": {"gathering": 43.0, "trusted": 43.0, "noise": 42.9}},
        "release": {"date": "2026-01-16", "number": "15.1"},
        "clan": {"acc": "CL00106", "id": "Csr_Rsm_clan"},
    }
}

# Two interleaved blocks: the consensus lines must be joined across them.
STOCKHOLM = (
    "# STOCKHOLM 1.0\n"
    "#=GF ID TwoAYGGAY\n"
    "seq1/1-8       GCCAUGGC\n"
    "#=GC SS_cons   <<<.__.>\n"
    "#=GC RF        GCCaUGGc\n"
    "\n"
    "seq1/1-8       AU\n"
    "#=GC SS_cons   >>\n"
    "#=GC RF        AU\n"
    "//\n"
)


class _FakeSession:
    """Serves canned responses by URL path, recording each request and the User-Agent it was built with."""

    def __init__(self, routes: dict[str, tuple[int, str]]):
        self.routes = routes
        self.urls: list[str] = []
        self.user_agent: str | None = None

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        status, text = self.routes.get(url, (404, '{"detail": "Not found."}'))
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = text.encode()
        return response

    def close(self):
        pass


def _serve(monkeypatch, routes):
    session = _FakeSession(routes)

    def build(**kwargs):
        session.user_agent = kwargs["user_agent"]
        return session

    monkeypatch.setattr(shared_data_models, "build_http_session", build)
    monkeypatch.setattr(shared_data_models, "_BACKOFF_SECONDS", 0.0)
    return session


# ============================================================================
# rfam-regions
# ============================================================================


def test_regions_normalize_minus_strand_and_read_header(monkeypatch):
    _serve(monkeypatch, {f"{BASE}/RF01731/regions": (200, REGIONS)})
    out = run_rfam_regions(RfamRegionsInput(family="RF01731"), RfamRegionsConfig())

    assert (out.accession, out.rfam_id, out.rfam_release) == ("RF01731", "TwoAYGGAY", "15.1")
    assert (out.total_regions, out.matched_regions, out.truncated) == (4, 4, False)
    minus = out.regions[1]
    assert (minus.start, minus.end, minus.strand) == (2296975, 2297101, "-")
    assert (out.regions[0].start, out.regions[0].end, out.regions[0].strand) == (1720658, 1720784, "+")


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"taxid": 216595}, 2),
        ({"species": "PROTEGENS"}, 2),
        ({"sequence_accession": "AM181176"}, 2),
        ({"sequence_accession": "AM181176.4"}, 2),
        ({"sequence_accession": "AM181176.3"}, 0),
        ({"taxid": 216595, "species": "protegens"}, 0),
        ({"species": "  "}, 4),
    ],
)
def test_regions_filters(monkeypatch, filters, expected):
    _serve(monkeypatch, {f"{BASE}/RF01731/regions": (200, REGIONS)})
    out = run_rfam_regions(RfamRegionsInput(family="RF01731", **filters), RfamRegionsConfig())
    assert out.matched_regions == expected
    assert out.total_regions == 4


def test_regions_truncate_after_filtering(monkeypatch):
    _serve(monkeypatch, {f"{BASE}/RF01731/regions": (200, REGIONS)})
    out = run_rfam_regions(RfamRegionsInput(family="RF01731"), RfamRegionsConfig(max_regions=3))
    assert (out.matched_regions, len(out.regions), out.truncated) == (4, 3, True)


@pytest.mark.parametrize("status", [413, 403])
def test_regions_relay_rfam_refusal_for_huge_families(monkeypatch, status):
    # Rfam answers 413 in practice; its API docs say 403.
    refusal = "'RF00005' has 5336050 regions, too many to return in a single response."
    _serve(monkeypatch, {f"{BASE}/RF00005/regions": (status, refusal)})
    with pytest.raises(ValueError, match="5336050 regions"):
        run_rfam_regions(RfamRegionsInput(family="RF00005"), RfamRegionsConfig())


def test_regions_family_name_is_url_escaped(monkeypatch):
    session = _serve(monkeypatch, {})
    with pytest.raises(ValueError):
        run_rfam_regions(RfamRegionsInput(family="a/b"), RfamRegionsConfig())
    assert session.urls == [f"{BASE}/a%2Fb/regions"]


def test_blank_family_rejected():
    with pytest.raises(ValidationError):
        RfamRegionsInput(family="   ")


def test_user_agent_names_a_hosted_caller(monkeypatch):
    monkeypatch.setenv(base_config.CLIENT_IDENTITY_ENV_VAR, "hosted-container")
    session = _serve(monkeypatch, _family_routes())
    config = RfamFamilyConfig.model_validate({"_proto_internal": {"client_identity": "user-123"}})
    run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), config)
    assert session.user_agent is not None and session.user_agent.endswith("(user-123)")


# ============================================================================
# rfam-family
# ============================================================================


def _family_routes() -> dict[str, tuple[int, str]]:
    return {
        f"{BASE}/TwoAYGGAY": (200, json.dumps(FAMILY)),
        f"{BASE}/RF01731/alignment/stockholm": (200, STOCKHOLM),
    }


def test_family_resolves_id_and_joins_consensus_blocks(monkeypatch):
    session = _serve(monkeypatch, _family_routes())
    out = run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), RfamFamilyConfig())

    assert session.urls[-1] == f"{BASE}/RF01731/alignment/stockholm"
    assert (out.accession, out.rfam_id, out.clan_id) == ("RF01731", "TwoAYGGAY", "Csr_Rsm_clan")
    assert out.consensus_structure == "<<<.__.>>>"
    assert out.consensus_sequence == "GCCaUGGcAU"


def test_family_strips_curation_semicolons(monkeypatch):
    _serve(monkeypatch, _family_routes())
    out = run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), RfamFamilyConfig())
    assert (out.rna_type, out.seed_source, out.comment) == ("Cis-reg", "Published; PMID:20230605", None)


def test_family_seed_alignment_on_request_and_export(tmp_path, monkeypatch):
    _serve(monkeypatch, _family_routes())
    out = run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), RfamFamilyConfig(include_seed_alignment=True))
    assert out.seed_alignment is not None and out.seed_alignment.rstrip() == STOCKHOLM.rstrip()
    out._export_output(tmp_path / "seed", "sto")
    assert (tmp_path / "seed.sto").read_text() == STOCKHOLM


def test_family_sto_export_needs_alignment(tmp_path, monkeypatch):
    _serve(monkeypatch, _family_routes())
    out = run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), RfamFamilyConfig())
    with pytest.raises(ValueError, match="include_seed_alignment"):
        out._export_output(tmp_path / "seed", "sto")


@pytest.mark.parametrize(
    ("run", "inputs", "config"),
    [
        (run_rfam_family, RfamFamilyInput(family="Nope"), RfamFamilyConfig()),
        (run_rfam_regions, RfamRegionsInput(family="Nope"), RfamRegionsConfig()),
    ],
)
def test_unknown_family(monkeypatch, run, inputs, config):
    _serve(monkeypatch, {})
    with pytest.raises(ValueError, match="No Rfam family matches 'Nope'"):
        run(inputs, config)


def test_family_alignment_without_structure_is_an_error(monkeypatch):
    routes = _family_routes()
    routes[f"{BASE}/RF01731/alignment/stockholm"] = (200, "# STOCKHOLM 1.0\nseq1 ACGU\n//\n")
    _serve(monkeypatch, routes)
    with pytest.raises(ValueError, match="SS_cons"):
        run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"), RfamFamilyConfig())


# ============================================================================
# Live (rfam.org)
# ============================================================================


@pytest.mark.integration
def test_live_regions_sbw25():
    out = run_rfam_regions(RfamRegionsInput(family="RF01731", taxid=216595))
    assert out.matched_regions >= 1
    top = max(out.regions, key=lambda r: r.bit_score)
    assert (top.sequence_accession, top.start, top.end, top.strand) == ("AM181176.4", 1720658, 1720784, "+")


@pytest.mark.integration
def test_live_family_by_id():
    out = run_rfam_family(RfamFamilyInput(family="TwoAYGGAY"))
    assert out.accession == "RF01731"
    assert len(out.consensus_structure) == len(out.consensus_sequence) > 0
