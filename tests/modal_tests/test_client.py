"""tests/test_client.py.

Client-side dispatch behaviour that has no container to prove it.

A container cannot return an exception, so conditions a caller must distinguish — an unprovisioned
asset in particular — travel as flags and are rebuilt here. These tests pin that reconstruction,
including against a container built before the flags carried their parts separately.
"""

import pytest


def test_a_missing_asset_is_rebuilt_rather_than_reported_as_a_bad_schema():
    """A container cannot send an exception, so the client rebuilds it from the flag it sends.

    Without this the payload simply fails validation — it has no item field — and the caller is
    told its result schema is malformed while the real cause survives only inside a string.
    """
    from proto_tools.modal.client import _validated_output
    from proto_tools.utils.tool_io import MissingAssetError

    payload = {
        "success": False,
        "errors": ["MissingAssetError: esm2: weights not provisioned"],
        "warnings": [],
        "missing_asset": True,
        "missing_asset_toolkit": "esm2",
        "missing_asset_kind": "weights",
    }

    with pytest.raises(MissingAssetError) as caught:
        _validated_output("esm2-score", payload)

    assert caught.value.toolkit == "esm2"
    assert caught.value.asset_kind == "weights"


def test_a_missing_asset_survives_a_container_built_before_the_structured_fields():
    """Deployed containers lag the client, so the flag alone still has to work."""
    from proto_tools.modal.client import _validated_output
    from proto_tools.utils.tool_io import MissingAssetError

    payload = {
        "success": False,
        "errors": ["MissingAssetError: esm2: weights not provisioned"],
        "warnings": [],
        "missing_asset": True,
    }

    with pytest.raises(MissingAssetError) as caught:
        _validated_output("esm2-score", payload)

    assert "weights not provisioned" in str(caught.value), "the cause is carried through in details"


def test_a_normal_result_is_untouched_by_the_missing_asset_check():
    """The check must not intercept an ordinary payload."""
    from proto_tools.modal.client import _raise_if_asset_missing

    _raise_if_asset_missing("esm2-score", {"success": True, "scores": []})
    _raise_if_asset_missing("esm2-score", {"success": False, "errors": ["something else broke"]})


def test_a_container_declares_itself_a_hosted_environment():
    """A container hosts the tool for someone else and cannot stage a large corpus on demand.

    proto-tools reads ``PROTO_IS_HOSTED_ENV`` to decide whether a config that would reach for one
    has to give way — an MSA search against a hundreds-of-gigabyte database being the case that
    matters. Declared here rather than sniffed from Modal there, so proto-tools stays unaware of
    which provider it is running on and the proto service can say the same thing later.
    """
    import os

    from proto_tools.modal.utils import dispatch_tool_call

    os.environ.pop("PROTO_IS_HOSTED_ENV", None)
    seen: dict[str, str | None] = {}

    def record(*_args, **_kwargs):
        """Stand in for a tool, capturing what the environment said while it ran."""
        seen["hosted"] = os.environ.get("PROTO_IS_HOSTED_ENV")

        class _Result:
            def model_dump(self):
                return {"tool_id": "x", "success": True}

        return _Result()

    try:
        dispatch_tool_call(record)
        assert seen["hosted"] == "1", "the tool must see the flag while it runs"
    finally:
        os.environ.pop("PROTO_IS_HOSTED_ENV", None)
