"""Tests for the structure prediction dispatch router."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from proto_tools.tools.structure_prediction.dispatch import SP_TOOL_MAP, predict_structures
from proto_tools.tools.structure_prediction.shared_data_models import StructurePredictionComplex


class _FakeConfig(BaseModel):
    temperature: float = 1.0


def _dispatch(tool_config=None, *, config_cls=_FakeConfig):
    """Run predict_structures with a fake tool map and return (run_func_mock, input_mock)."""
    mock_run = MagicMock()
    mock_input = MagicMock()
    fake_map = {"fake": {"config": config_cls, "input": mock_input, "run_func": mock_run}}
    cpx = StructurePredictionComplex(chains=["MVLSPADKTN"])
    with patch.dict(SP_TOOL_MAP, fake_map, clear=True):
        predict_structures(cpx, toolkit="fake", tool_config=tool_config)
    return mock_run, mock_input


def test_unknown_tool_raises():
    cpx = StructurePredictionComplex(chains=["MVLSPADKTN"])
    with pytest.raises(ValueError, match="Unknown structure prediction tool"):
        predict_structures(cpx, toolkit="nonexistent_tool")


def test_single_complex_normalized_to_list():
    _, mock_input = _dispatch()
    complexes_arg = mock_input.call_args.kwargs.get("complexes", mock_input.call_args[1].get("complexes"))
    assert isinstance(complexes_arg, list)


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
