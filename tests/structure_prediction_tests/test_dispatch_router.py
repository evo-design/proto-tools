"""Tests for the structure prediction dispatch router."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.dispatch import SP_TOOL_MAP, predict_structures
from proto_tools.tools.structure_prediction.shared_data_models import Complex, ComplexMSAs


class _FakeConfig(BaseModel):
    temperature: float = 1.0


def _dispatch(tool_config=None, *, config_cls=_FakeConfig, msas=None):
    """Run predict_structures with a fake tool map and return (run_func_mock, input_mock)."""
    mock_run = MagicMock()
    mock_input = MagicMock()
    fake_map = {"fake": {"config": config_cls, "input": mock_input, "run_func": mock_run}}
    cpx = Complex(chains=["MVLSPADKTN"])
    with patch.dict(SP_TOOL_MAP, fake_map, clear=True):
        predict_structures(cpx, toolkit="fake", tool_config=tool_config, msas=msas)
    return mock_run, mock_input


def test_unknown_tool_raises():
    cpx = Complex(chains=["MVLSPADKTN"])
    with pytest.raises(ValueError, match=r"predict_structures: unknown toolkit"):
        predict_structures(cpx, toolkit="nonexistent_tool")


def test_single_complex_normalized_to_list():
    _, mock_input = _dispatch()
    complexes_arg = mock_input.call_args.kwargs.get("complexes", mock_input.call_args[1].get("complexes"))
    assert isinstance(complexes_arg, list)


def test_msas_default_none():
    _, mock_input = _dispatch()
    assert mock_input.call_args.kwargs["msas"] is None


def test_msas_forwarded_to_input():
    # A target-only MSA: chain 1 conditioned, the binder (chain 0) omitted -> single-sequence.
    msas = [ComplexMSAs(per_chain={1: MSA(aligned_sequences=["MVLSPADKTN", "MVLSPADKTN"])}, paired=False)]
    _, mock_input = _dispatch(msas=msas)
    assert mock_input.call_args.kwargs["msas"] is msas


def test_single_msas_normalized_to_list():
    # A bare ComplexMSAs is wrapped to a one-element list, mirroring the single-Complex convenience.
    single = ComplexMSAs(per_chain={1: MSA(aligned_sequences=["MVLSPADKTN", "MVLSPADKTN"])}, paired=False)
    _, mock_input = _dispatch(msas=single)
    assert mock_input.call_args.kwargs["msas"] == [single]


@pytest.mark.parametrize(
    "tool_config,expected_temp",
    [
        ({"temperature": 0.5}, 0.5),
        (None, 1.0),
    ],
    ids=["dict-config", "none-default"],
)
def test_config_conversion(tool_config, expected_temp):
    mock_run, _ = _dispatch(tool_config)
    actual_config = mock_run.call_args[0][1]
    assert isinstance(actual_config, _FakeConfig)
    assert actual_config.temperature == expected_temp


def test_config_type_mismatch_raises():
    class _WrongConfig(BaseModel):
        pass

    with pytest.raises(ValueError, match="doesn't match"):
        _dispatch(tool_config=_WrongConfig())
