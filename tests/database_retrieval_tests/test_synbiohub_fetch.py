"""Tests for SynBioHub design retrieval."""

import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from Bio import SeqIO
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    SynBioHubFetchConfig,
    SynBioHubFetchInput,
    SynBioHubFetchOutput,
    SynBioHubPart,
    run_synbiohub_fetch,
)
from proto_tools.tools.database_retrieval.synbiohub import synbiohub_fetch

DESIGN_URI = "https://api.synbiohub.org/public/test/device/1"
PUBLIC = "https://synbiohub.org/public/test"

# A device of an RBS, a reverse-strand CDS, and a split terminator, identified on the public host
DEVICE_SBOL = f"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dcterms="http://purl.org/dc/terms/"
         xmlns:sbol="http://sbols.org/v2#">
  <sbol:ComponentDefinition rdf:about="{PUBLIC}/device/1">
    <sbol:displayId>device</sbol:displayId>
    <dcterms:title>Test device</dcterms:title>
    <dcterms:description>RBS, CDS, and terminator</dcterms:description>
    <dcterms:created>2005-05-30T11:00:00Z</dcterms:created>
    <dcterms:modified>2015-08-31T04:07:34Z</dcterms:modified>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#DnaRegion"/>
    <sbol:type rdf:resource="http://identifiers.org/so/SO:0000988"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000804"/>
    <sbol:role rdf:resource="http://wiki.synbiohub.org/wiki/Terms/igem#partType/Reporter"/>
    <sbol:component>
      <sbol:Component rdf:about="{PUBLIC}/device/c_rbs/1">
        <sbol:definition rdf:resource="{PUBLIC}/rbs/1"/>
      </sbol:Component>
    </sbol:component>
    <sbol:component>
      <sbol:Component rdf:about="{PUBLIC}/device/c_cds/1">
        <sbol:definition rdf:resource="{PUBLIC}/cds/1"/>
      </sbol:Component>
    </sbol:component>
    <sbol:sequenceAnnotation>
      <sbol:SequenceAnnotation rdf:about="{PUBLIC}/device/a_rbs/1">
        <sbol:displayId>a_rbs</sbol:displayId>
        <sbol:location>
          <sbol:Range rdf:about="{PUBLIC}/device/a_rbs/range/1">
            <sbol:start>1</sbol:start>
            <sbol:end>4</sbol:end>
            <sbol:orientation rdf:resource="http://sbols.org/v2#inline"/>
          </sbol:Range>
        </sbol:location>
        <sbol:component rdf:resource="{PUBLIC}/device/c_rbs/1"/>
      </sbol:SequenceAnnotation>
    </sbol:sequenceAnnotation>
    <sbol:sequenceAnnotation>
      <sbol:SequenceAnnotation rdf:about="{PUBLIC}/device/a_cds/1">
        <sbol:displayId>a_cds</sbol:displayId>
        <dcterms:title>reporter</dcterms:title>
        <sbol:location>
          <sbol:Range rdf:about="{PUBLIC}/device/a_cds/range/1">
            <sbol:start>5</sbol:start>
            <sbol:end>10</sbol:end>
            <sbol:orientation rdf:resource="http://sbols.org/v2#reverseComplement"/>
          </sbol:Range>
        </sbol:location>
        <sbol:component rdf:resource="{PUBLIC}/device/c_cds/1"/>
      </sbol:SequenceAnnotation>
    </sbol:sequenceAnnotation>
    <sbol:sequenceAnnotation>
      <sbol:SequenceAnnotation rdf:about="{PUBLIC}/device/a_term/1">
        <sbol:displayId>a_term</sbol:displayId>
        <sbol:role rdf:resource="http://identifiers.org/so/SO:0000141"/>
        <sbol:location>
          <sbol:Range rdf:about="{PUBLIC}/device/a_term/r1/1"><sbol:start>11</sbol:start><sbol:end>12</sbol:end></sbol:Range>
        </sbol:location>
        <sbol:location>
          <sbol:Range rdf:about="{PUBLIC}/device/a_term/r2/1"><sbol:start>13</sbol:start><sbol:end>14</sbol:end></sbol:Range>
        </sbol:location>
      </sbol:SequenceAnnotation>
    </sbol:sequenceAnnotation>
    <sbol:sequenceAnnotation>
      <sbol:SequenceAnnotation rdf:about="{PUBLIC}/device/a_cut/1">
        <sbol:displayId>a_cut</sbol:displayId>
        <sbol:location>
          <sbol:Cut rdf:about="{PUBLIC}/device/a_cut/cut/1"><sbol:at>4</sbol:at></sbol:Cut>
        </sbol:location>
      </sbol:SequenceAnnotation>
    </sbol:sequenceAnnotation>
    <sbol:sequence rdf:resource="{PUBLIC}/device_sequence/1"/>
  </sbol:ComponentDefinition>
  <sbol:Sequence rdf:about="{PUBLIC}/device_sequence/1">
    <sbol:elements>aaaggatgca
tgcc</sbol:elements>
    <sbol:encoding rdf:resource="http://www.chem.qmul.ac.uk/iubmb/misc/naseq.html"/>
  </sbol:Sequence>
  <sbol:ComponentDefinition rdf:about="{PUBLIC}/rbs/1">
    <sbol:displayId>rbs</sbol:displayId>
    <dcterms:title>RBS part</dcterms:title>
    <dcterms:created>2004-01-02T00:00:00Z</dcterms:created>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#DnaRegion"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000139"/>
  </sbol:ComponentDefinition>
  <sbol:ComponentDefinition rdf:about="{PUBLIC}/cds/1">
    <sbol:displayId>cds</sbol:displayId>
    <dcterms:title>CDS part</dcterms:title>
    <sbol:type rdf:resource="http://www.biopax.org/release/biopax-level3.owl#DnaRegion"/>
    <sbol:role rdf:resource="http://identifiers.org/so/SO:0000316"/>
  </sbol:ComponentDefinition>
</rdf:RDF>
"""

COLLECTION_SBOL = f"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:sbol="http://sbols.org/v2#">
  <sbol:Collection rdf:about="{PUBLIC}/device/1"><sbol:displayId>device</sbol:displayId></sbol:Collection>
</rdf:RDF>
"""


def _mock_session(
    monkeypatch, *, content: str = DEVICE_SBOL, status_code: int = 200, versions: list[str] | None = None
) -> MagicMock:
    response = MagicMock(content=content.encode(), encoding="utf-8", status_code=status_code)
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": {"bindings": [{"s": {"value": uri}} for uri in versions or []]}}
    session = MagicMock()
    session.get.return_value = response
    session.post.return_value = MagicMock(status_code=200, text="session-token\n")
    monkeypatch.setattr(synbiohub_fetch, "build_http_session", lambda **kwargs: session)
    return session


def _fetch(**config) -> SynBioHubFetchOutput:
    config = {"synbiohub_email": None, "synbiohub_password": None, **config}
    return run_synbiohub_fetch(SynBioHubFetchInput(identifier=DESIGN_URI), SynBioHubFetchConfig(**config))


LOGIN = {"synbiohub_email": "me@example.org", "synbiohub_password": "private-password"}


# ── Validation ────────────────────────────────────────────────────────────────


def test_identifier_rejects_query_parameters():
    with pytest.raises(ValidationError, match="query parameters"):
        SynBioHubFetchInput(identifier="https://synbiohub.org/public/igem/BBa_E0040/1?token=secret")


@pytest.mark.parametrize(
    ("identifier", "normalized"),
    [
        ("BBa_I13504", "BBa_I13504"),
        ("  https://synbiohub.org/public/igem/BBa_I13504/1/ ", "https://synbiohub.org/public/igem/BBa_I13504/1"),
        ("synbiohub.org/public/igem/BBa_I13504/1", "https://synbiohub.org/public/igem/BBa_I13504/1"),
    ],
)
def test_identifier_normalization(identifier, normalized):
    assert SynBioHubFetchInput(identifier=identifier).identifier == normalized


API = "https://api.synbiohub.org"


@pytest.mark.parametrize(
    ("identifier", "download_uri"),
    [
        (f"{API}/public/igem/BBa_I13504/1", f"{API}/public/igem/BBa_I13504/1"),
        ("http://api.synbiohub.org/public/igem/BBa_I13504/1", f"{API}/public/igem/BBa_I13504/1"),
        ("https://synbiohub.org/public/igem/BBa_I13504/1", f"{API}/public/igem/BBa_I13504/1"),
        ("http://www.synbiohub.org/public/igem/BBa_I13504", f"{API}/public/igem/BBa_I13504"),
        ("https://SynBioHub.org/user/alice/designs/device/1", f"{API}/user/alice/designs/device/1"),
        ("BBa_I13504", f"{API}/public/igem/BBa_I13504"),
        ("bba_i13504", f"{API}/public/igem/BBa_I13504"),
        ("pSB1C3", f"{API}/public/igem/pSB1C3"),
        ("https://parts.igem.org/Part:BBa_I13504", f"{API}/public/igem/BBa_I13504"),
        ("http://parts.igem.org/Part:BBa_I13504:Design", f"{API}/public/igem/BBa_I13504"),
        ("https://registry.igem.org/parts/bba-i13504", f"{API}/public/igem/BBa_I13504"),
        ("https://registry.igem.org/parts/BBa_I13504", f"{API}/public/igem/BBa_I13504"),
    ],
)
def test_identifier_resolves_to_a_registry_download(identifier, download_uri):
    identifier = SynBioHubFetchInput(identifier=identifier).identifier
    assert synbiohub_fetch._download_uri(identifier, API) == download_uri


def test_igem_link_without_a_part_is_refused():
    with pytest.raises(ValueError, match="does not link to an iGEM part"):
        synbiohub_fetch._download_uri("https://parts.igem.org/Main_Page", API)


def test_fetch_rejects_cross_origin_object():
    with pytest.raises(ValueError, match="same scheme and host"):
        run_synbiohub_fetch(
            SynBioHubFetchInput(identifier="https://example.org/public/igem/BBa_E0040/1"),
            SynBioHubFetchConfig(registry_url="https://synbiohub.org"),
        )


def test_website_links_need_the_public_registry():
    with pytest.raises(ValueError, match="same scheme and host"):
        run_synbiohub_fetch(
            SynBioHubFetchInput(identifier=f"{PUBLIC}/device/1"),
            SynBioHubFetchConfig(registry_url="https://synbiohub.example.org"),
        )


# ── Retrieval ─────────────────────────────────────────────────────────────────


def test_logs_in_checks_type_downloads_and_logs_out_without_exposing_credentials(monkeypatch):
    session = _mock_session(monkeypatch)

    output = _fetch(**LOGIN)

    login, logout = session.post.call_args_list
    assert login.args[0] == f"{API}/login"
    assert login.kwargs["data"] == {"email": "me@example.org", "password": "private-password"}
    assert [call.args[0] for call in session.get.call_args_list] == [f"{DESIGN_URI}/sbolnr", f"{DESIGN_URI}/sbol"]
    assert all(call.kwargs["headers"]["X-authorization"] == "session-token" for call in session.get.call_args_list)
    assert logout.args[0] == f"{API}/logout"
    assert logout.kwargs["headers"]["X-authorization"] == "session-token"
    assert output.source_url == f"{DESIGN_URI}/sbol"
    assert all(secret not in output.model_dump_json() for secret in ("private-password", "session-token"))


def test_redirect_is_refused(monkeypatch):
    _mock_session(monkeypatch, status_code=302)
    with pytest.raises(ValueError, match="redirected"):
        _fetch()


def test_website_link_downloads_from_the_api_host(monkeypatch):
    session = _mock_session(monkeypatch)

    output = run_synbiohub_fetch(SynBioHubFetchInput(identifier=f"{PUBLIC}/device/1"), SynBioHubFetchConfig())

    assert [call.args[0] for call in session.get.call_args_list] == [f"{DESIGN_URI}/sbolnr", f"{DESIGN_URI}/sbol"]
    assert output.identifier == f"{PUBLIC}/device/1"
    assert output.design.uri == f"{PUBLIC}/device/1"


def test_unversioned_identifier_resolves_its_version_before_any_sbol_download(monkeypatch):
    session = _mock_session(monkeypatch, versions=[f"{PUBLIC}/device/1"])

    output = run_synbiohub_fetch(SynBioHubFetchInput(identifier=f"{PUBLIC}/device"), SynBioHubFetchConfig())

    calls = session.get.call_args_list
    assert [call.args[0] for call in calls] == [f"{API}/sparql", f"{DESIGN_URI}/sbolnr", f"{DESIGN_URI}/sbol"]
    assert f"persistentIdentity> <{PUBLIC}/device>" in calls[0].kwargs["params"]["query"]
    assert output.source_url == f"{DESIGN_URI}/sbol"


def test_unversioned_identifier_with_several_versions_is_refused(monkeypatch):
    session = _mock_session(monkeypatch, versions=[f"{PUBLIC}/device/1", f"{PUBLIC}/device/2"])
    with pytest.raises(ValueError, match="several versions"):
        run_synbiohub_fetch(SynBioHubFetchInput(identifier=f"{PUBLIC}/device"), SynBioHubFetchConfig())
    assert [call.args[0] for call in session.get.call_args_list] == [f"{API}/sparql"]


def test_missing_object_on_the_registry_is_reported(monkeypatch):
    _mock_session(monkeypatch, status_code=404)
    with pytest.raises(ValueError, match="no object at"):
        _fetch()


def test_missing_login_is_reported(monkeypatch):
    _mock_session(monkeypatch, status_code=401)
    with pytest.raises(ValueError, match="requires a login; set SYNBIOHUB_EMAIL and SYNBIOHUB_PASSWORD"):
        _fetch()


def test_rejected_login_is_reported(monkeypatch):
    session = _mock_session(monkeypatch)
    session.post.return_value = MagicMock(status_code=401, text="Your e-mail address was not recognized.")
    with pytest.raises(ValueError, match="rejected the email or password"):
        _fetch(**LOGIN)
    assert session.post.call_count == 1


def test_logs_out_when_the_fetch_fails(monkeypatch):
    session = _mock_session(monkeypatch, status_code=404)
    with pytest.raises(ValueError, match="no object"):
        _fetch(**LOGIN)
    assert session.post.call_args_list[-1].args[0] == f"{API}/logout"


def test_object_that_is_not_a_component_definition_is_refused_before_the_recursive_download(monkeypatch):
    session = _mock_session(monkeypatch, content=COLLECTION_SBOL)
    with pytest.raises(ValueError, match="is a Collection, not a ComponentDefinition"):
        _fetch()
    assert [call.args[0] for call in session.get.call_args_list] == [f"{DESIGN_URI}/sbolnr"]


def test_missing_object_is_refused(monkeypatch):
    _mock_session(monkeypatch, content=COLLECTION_SBOL.replace("device/1", "other/1"))
    with pytest.raises(ValueError, match="is not a ComponentDefinition"):
        _fetch()


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parses_design_across_download_and_public_hosts(monkeypatch):
    _mock_session(monkeypatch)

    design = _fetch().design

    assert design.uri == f"{PUBLIC}/device/1"
    assert (design.display_id, design.name, design.molecule) == ("device", "Test device", "dna")
    assert design.roles == ["SO:0000804"]
    assert design.sequence == "AAAGGATGCATGCC"
    assert design.topology == "circular"
    assert design.modified == datetime(2015, 8, 31, 4, 7, 34, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2015-08-31T04:07:34Z", datetime(2015, 8, 31, 4, 7, 34, tzinfo=UTC)),
        (" 2015-08-31T04:07:34 ", datetime(2015, 8, 31, 4, 7, 34, tzinfo=UTC)),
        ("2015-08-31T06:07:34+02:00", datetime(2015, 8, 31, 4, 7, 34, tzinfo=UTC)),
        ("2015-08-31T04:07:34.1234Z", datetime(2015, 8, 31, 4, 7, 34, 123400, tzinfo=UTC)),
        ("2015-08-31", datetime(2015, 8, 31, tzinfo=UTC)),
        ("last Tuesday", None),
        ("", None),
    ],
)
def test_timestamps_parse_to_utc(value, expected):
    element = ET.Element("part")
    ET.SubElement(element, "{http://purl.org/dc/terms/}modified").text = value

    parsed = synbiohub_fetch._timestamp(element, "modified")

    assert parsed == expected
    assert parsed is None or parsed.tzinfo is UTC


def test_malformed_modification_time_falls_back_to_creation_time(monkeypatch):
    _mock_session(monkeypatch, content=DEVICE_SBOL.replace("2015-08-31T04:07:34Z", "unknown"))

    output = _fetch()

    assert output.design.modified == datetime(2005, 5, 30, 11, tzinfo=UTC)
    assert '"modified":"2005-05-30T11:00:00Z"' in output.model_dump_json()


def test_parses_features_with_locations_and_inherited_roles(monkeypatch):
    _mock_session(monkeypatch)

    features = {feature.display_id: feature for feature in _fetch().design.features}

    assert set(features) == {"a_rbs", "a_cds", "a_term"}
    rbs, cds, term = features["a_rbs"], features["a_cds"], features["a_term"]
    assert (rbs.name, rbs.roles, rbs.subpart_uri) == ("RBS part", ["SO:0000139"], f"{PUBLIC}/rbs/1")
    assert [(loc.start, loc.end, loc.strand) for loc in rbs.locations] == [(1, 4, "+")]
    assert (cds.name, cds.roles) == ("reporter", ["SO:0000316"])
    assert [(loc.start, loc.end, loc.strand) for loc in cds.locations] == [(5, 10, "-")]
    assert (term.roles, term.subpart_uri) == (["SO:0000141"], None)
    assert [(loc.start, loc.end) for loc in term.locations] == [(11, 12), (13, 14)]


def test_subparts_list_every_other_definition(monkeypatch):
    _mock_session(monkeypatch)

    subparts = {part.uri: part for part in _fetch().subparts}

    assert set(subparts) == {f"{PUBLIC}/rbs/1", f"{PUBLIC}/cds/1"}
    assert subparts[f"{PUBLIC}/cds/1"].roles == ["SO:0000316"]
    assert subparts[f"{PUBLIC}/cds/1"].sequence is None
    assert subparts[f"{PUBLIC}/cds/1"].topology is None
    assert subparts[f"{PUBLIC}/rbs/1"].modified == datetime(2004, 1, 2, tzinfo=UTC)


# ── Export ────────────────────────────────────────────────────────────────────


def test_sbol_export_is_verbatim(tmp_path, monkeypatch):
    _mock_session(monkeypatch)

    _fetch().export("device", export_path=tmp_path, file_format="sbol")

    assert (tmp_path / "device.sbol").read_text(encoding="utf-8") == DEVICE_SBOL


def test_fasta_export(tmp_path, monkeypatch):
    _mock_session(monkeypatch)

    _fetch().export("device", export_path=tmp_path, file_format="fasta")

    assert (tmp_path / "device.fasta").read_text() == ">device RBS, CDS, and terminator\nAAAGGATGCATGCC\n"


def test_fasta_header_stays_on_one_line(tmp_path, monkeypatch):
    description = "<dcterms:description>RBS, CDS, and terminator</dcterms:description>"
    _mock_session(monkeypatch, content=DEVICE_SBOL.replace(description, description.replace(", CDS,", ",\n  CDS,\t")))

    output = _fetch()
    output.export("device", export_path=tmp_path, file_format="fasta")

    assert "\n" in output.design.description
    assert (tmp_path / "device.fasta").read_text() == ">device RBS, CDS, and terminator\nAAAGGATGCATGCC\n"


def test_genbank_export_maps_roles_and_coordinates(tmp_path, monkeypatch):
    _mock_session(monkeypatch)

    _fetch().export("device", export_path=tmp_path, file_format="genbank")

    record = SeqIO.read(tmp_path / "device.gb", "genbank")
    assert str(record.seq) == "AAAGGATGCATGCC"
    assert (record.annotations["topology"], record.annotations["date"]) == ("circular", "31-AUG-2015")
    features = [(f.type, f.qualifiers["label"][0], str(f.location)) for f in record.features]
    assert features == [
        ("RBS", "RBS part", "[0:4](+)"),
        ("CDS", "reporter", "[4:10](-)"),
        ("terminator", "a_term", "join{[10:12](+), [12:14](+)}"),
    ]


def test_sequence_exports_need_a_sequence(tmp_path):
    part = SynBioHubPart(uri=f"{PUBLIC}/protein/1", display_id="protein", molecule="protein")
    output = SynBioHubFetchOutput(
        identifier=DESIGN_URI,
        registry_url="https://api.synbiohub.org",
        source_url=f"{DESIGN_URI}/sbol",
        design=part,
        sbol="",
    )

    with pytest.raises(ValueError, match="no sequence"):
        output.export("protein", export_path=tmp_path, file_format="fasta")


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("SYNBIOHUB_EMAIL") and os.environ.get("SYNBIOHUB_PASSWORD")),
    reason="SYNBIOHUB_EMAIL and SYNBIOHUB_PASSWORD are not set",
)
def test_live_gfp_cassette():
    output = run_synbiohub_fetch(synbiohub_fetch.example_input(), SynBioHubFetchConfig())

    assert output.design.display_id == "BBa_I13504"
    assert len(output.design.sequence or "") == 875
    ranges = sorted((f.locations[0].start, f.locations[0].end) for f in output.design.features)
    assert ranges == [(1, 12), (19, 738), (747, 826), (835, 875)]
