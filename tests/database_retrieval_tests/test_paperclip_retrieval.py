"""Tests for optional Paperclip literature retrieval."""

import importlib.util
import json
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    PaperclipDocumentConfig,
    PaperclipDocumentInput,
    PaperclipSearchConfig,
    PaperclipSearchInput,
    run_paperclip_document,
    run_paperclip_search,
)
from proto_tools.tools.database_retrieval.paperclip import paperclip_retrieval
from proto_tools.tools.tool_registry import ToolRegistry

_STANDALONE_DIR = Path(paperclip_retrieval.__file__).with_name("standalone")
_STANDALONE_SPEC = importlib.util.spec_from_file_location("paperclip_standalone", _STANDALONE_DIR / "run.py")
assert _STANDALONE_SPEC is not None and _STANDALONE_SPEC.loader is not None
paperclip_standalone = importlib.util.module_from_spec(_STANDALONE_SPEC)
_STANDALONE_SPEC.loader.exec_module(paperclip_standalone)


def test_search_normalizes_citable_hits(monkeypatch):
    def dispatch(toolkit, payload, **kwargs):
        assert toolkit == "paperclip"
        assert payload["operation"] == "search"
        assert payload["query"] == "CAR T"
        assert kwargs["config"].limit == 10
        return {
            "result_id": "s_fixture",
            "papers": [
                {
                    "document_id": "PMC11754230",
                    "title": "CAR-T cell therapy",
                    "source": "pmc",
                    "abstract": "An abstract.",
                    "chunk_text": "A matched passage.",
                    "pmc_id": "PMC11754230",
                    "doi": "10.1/example",
                    "citation_canonical_work": {"pmid": "39850899"},
                    "rerank_score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(paperclip_retrieval.ToolInstance, "dispatch", dispatch)
    output = run_paperclip_search(PaperclipSearchInput(query="CAR T"), PaperclipSearchConfig())
    hit = output.hits[0]
    assert output.result_id == "s_fixture"
    assert hit.document_id == "PMC11754230"
    assert hit.pmid == "39850899"
    assert hit.access_kind == "abstract"
    assert hit.provenance.provider == "Paperclip"


def test_document_preserves_exact_full_text_and_checksum(monkeypatch):
    def dispatch(toolkit, payload, **kwargs):
        assert toolkit == "paperclip"
        assert payload["operation"] == "document"
        assert payload["document_id"] == "PMC11754230"
        assert kwargs["config"].include_full_text is True
        return {
            "metadata_output": json.dumps({"source": "pmc", "license": "CC BY", "doi": "10.1/example"}),
            "content_exit_code": 0,
            "content_output": "L1 First line\nL2 Second line\n",
            "wc_exit_code": 0,
            "wc_output": "2 6 29 /papers/PMC11754230/content.lines",
        }

    monkeypatch.setattr(paperclip_retrieval.ToolInstance, "dispatch", dispatch)
    output = run_paperclip_document(PaperclipDocumentInput(document_id="PMC11754230"), PaperclipDocumentConfig())
    document = output.document
    assert document.access_kind == "full_text"
    assert document.content == "L1 First line\nL2 Second line\n"
    assert document.metadata_artifact.matches_content(
        json.dumps({"source": "pmc", "license": "CC BY", "doi": "10.1/example"}).encode()
    )
    assert document.artifact is not None
    assert document.artifact.matches_content(document.content.encode())
    assert document.provenance.license == "CC BY"


def test_document_id_cannot_escape_virtual_papers_directory():
    with pytest.raises(ValidationError, match="without a path"):
        PaperclipDocumentInput(document_id="../../credentials.json")


def test_canonical_standalone_environment_is_self_contained():
    required = {
        "env_vars.txt",
        "python_version.txt",
        "requirements.txt",
        "run.py",
        "setup.sh",
    }
    assert required <= {path.name for path in _STANDALONE_DIR.iterdir()}
    setup = (_STANDALONE_DIR / "setup.sh").read_text(encoding="utf-8")
    assert 'PAPERCLIP_VERSION="0.7.52"' in setup
    assert 'PAPERCLIP_SHA256="d25836294c63273f6cebe7d114fbefb2c45766d8cdb582277dd4ece5adb7a9b9"' in setup
    assert "assert version(" not in setup
    env_vars = (_STANDALONE_DIR / "env_vars.txt").read_text(encoding="utf-8")
    assert "PAPERCLIP_API_KEY" in env_vars


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
def test_setup_installs_paperclip_into_a_clean_tool_environment(tmp_path):
    """The tool setup provisions both uv and Paperclip without ambient installs."""
    tool_env = tmp_path / "paperclip-env"
    venv.EnvBuilder(with_pip=True).create(tool_env)
    python = tool_env / "bin" / "python"
    helper_dir = Path(__file__).parents[2] / "proto_tools" / "utils" / "standalone_helpers_source"
    clean_env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{helper_dir}:/usr/bin:/bin",
        "PYTHON_EXE": str(python),
        "VENV_PATH": str(tool_env),
    }
    (tmp_path / "home").mkdir()

    absent = subprocess.run(
        [str(python), "-I", "-c", "import importlib.util; assert importlib.util.find_spec('gxl_paperclip') is None"],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env,
    )
    assert absent.returncode == 0, absent.stderr

    setup = subprocess.run(
        ["bash", str(_STANDALONE_DIR / "setup.sh")],
        cwd=_STANDALONE_DIR,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env=clean_env,
    )
    assert setup.returncode == 0, setup.stdout[-2000:] + setup.stderr[-2000:]

    installed = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; import gxl_paperclip; print(version('gxl-paperclip'))",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env,
    )
    assert installed.returncode == 0, installed.stderr
    assert installed.stdout.strip() == "0.7.52"


@pytest.mark.parametrize("key", ["paperclip-literature-search", "paperclip-literature-document"])
def test_registry_routes_paperclip_through_managed_environment(key):
    spec = ToolRegistry.get(key)
    assert spec.has_standalone_env is True
    assert spec.local_cpu is False
    assert spec.local_only is None


def test_standalone_client_uses_official_environment_auth(monkeypatch):
    captured = {}
    expected_client = object()

    class _PaperclipClient:
        @classmethod
        def from_env(cls, **kwargs):
            captured.update(kwargs)
            return expected_client

    client_module = SimpleNamespace(PaperclipClient=_PaperclipClient)

    def import_module(name):
        assert name == "gxl_paperclip"
        return client_module

    monkeypatch.setattr(paperclip_standalone.importlib, "import_module", import_module)

    client = paperclip_standalone._client(
        {
            "paperclip_api_key": None,
            "paperclip_base_url": "https://paperclip.example",
            "timeout_seconds": 30,
        }
    )

    assert client is expected_client
    assert captured["base_url"] == "https://paperclip.example"
    assert captured["timeout"] == 30


def test_standalone_client_uses_explicit_api_key(monkeypatch):
    captured = {}
    expected_client = object()

    class _APIKeyAuth:
        def __init__(self, value):
            captured["api_key"] = value

    class _PaperclipClient:
        def __new__(cls, **kwargs):
            captured.update(kwargs)
            return expected_client

    client_module = SimpleNamespace(PaperclipClient=_PaperclipClient, APIKeyAuth=_APIKeyAuth)
    monkeypatch.setattr(paperclip_standalone.importlib, "import_module", lambda name: client_module)

    client = paperclip_standalone._client(
        {
            "paperclip_api_key": "gxl_test_key",
            "paperclip_base_url": None,
            "timeout_seconds": 45,
        }
    )

    assert client is expected_client
    assert captured["api_key"] == "gxl_test_key"
    assert isinstance(captured["auth"], _APIKeyAuth)
    assert captured["timeout"] == 45


def test_standalone_client_retains_legacy_library_path_fallback(monkeypatch, tmp_path):
    expected_client = object()
    calls = []

    class _PaperclipClient:
        @classmethod
        def from_env(cls, **kwargs):
            return expected_client

    client_module = SimpleNamespace(PaperclipClient=_PaperclipClient)

    def import_module(name):
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError("No module named 'gxl_paperclip'", name="gxl_paperclip")
        assert str(tmp_path) in paperclip_standalone.sys.path
        return client_module

    monkeypatch.setattr(paperclip_standalone.importlib, "import_module", import_module)

    client = paperclip_standalone._client(
        {
            "paperclip_library_path": str(tmp_path),
            "paperclip_api_key": None,
            "paperclip_base_url": None,
            "timeout_seconds": 30,
        }
    )

    assert client is expected_client
    assert calls == ["gxl_paperclip", "gxl_paperclip"]
    assert str(tmp_path) not in paperclip_standalone.sys.path


def test_standalone_client_does_not_mask_missing_sdk_dependency(monkeypatch, tmp_path):
    def import_module(name):
        raise ModuleNotFoundError("No module named 'rookiepy'", name="rookiepy")

    monkeypatch.setattr(paperclip_standalone.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="rookiepy"):
        paperclip_standalone._client(
            {
                "paperclip_library_path": str(tmp_path),
                "paperclip_api_key": None,
                "paperclip_base_url": None,
                "timeout_seconds": 30,
            }
        )

    assert str(tmp_path) not in paperclip_standalone.sys.path


def test_standalone_rejects_unknown_operation_before_loading_client(monkeypatch):
    monkeypatch.setattr(
        paperclip_standalone,
        "_client",
        lambda config: pytest.fail("unknown operation must not initialize an authenticated client"),
    )

    with pytest.raises(ValueError, match="Unsupported Paperclip operation"):
        paperclip_standalone.run_paperclip({"operation": "delete", "config": {}})


def test_standalone_search_calls_typed_client(monkeypatch):
    captured = {}

    class _Client:
        def search(self, query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return SimpleNamespace(exit_code=0, result_id="s_fixture", papers=[{"document_id": "PMC1"}])

    monkeypatch.setattr(paperclip_standalone, "_client", lambda config: _Client())
    result = paperclip_standalone.run_paperclip(
        {
            "operation": "search",
            "query": "protein design",
            "config": {
                "sources": ["pmc", "abstracts"],
                "limit": 7,
                "exact": False,
                "since": None,
                "timeout_seconds": 30,
            },
        }
    )

    assert result == {"result_id": "s_fixture", "papers": [{"document_id": "PMC1"}]}
    assert captured["query"] == "protein design"
    assert captured["source"] == "pmc,abstracts"
    assert captured["limit"] == 7


def test_standalone_document_returns_remote_completeness_evidence(monkeypatch):
    class _Client:
        @property
        def papers(self):
            return self

        def cat(self, path, **kwargs):
            if path.endswith("meta.json"):
                return SimpleNamespace(exit_code=0, output='{"source":"pmc"}')
            return SimpleNamespace(exit_code=0, output="L1 text\n")

        def execute(self, command, args, **kwargs):
            assert command == "wc"
            return SimpleNamespace(exit_code=0, output="1 2 8 /papers/PMC1/content.lines")

    monkeypatch.setattr(paperclip_standalone, "_client", lambda config: _Client())
    result = paperclip_standalone.run_paperclip(
        {
            "operation": "document",
            "document_id": "PMC1",
            "config": {"include_full_text": True, "timeout_seconds": 30},
        }
    )

    assert result["metadata_output"] == '{"source":"pmc"}'
    assert result["content_output"] == "L1 text\n"
    assert result["wc_output"].startswith("1 2 8")
