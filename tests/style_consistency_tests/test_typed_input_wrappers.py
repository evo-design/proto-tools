"""tests/style_consistency_tests/test_typed_input_wrappers.py.

Approved typed-input wrappers must coerce bare inputs at construction time.

Tools type fields like ``target_pdb: Structure`` and trust that callers can
pass a path, content string, dict, or wrapper instance — all coerced uniformly
with content embedded so the value is wire-safe (no caller-side path leaks).
The contract for every approved wrapper: a Pydantic ``model_validator(mode='before')``
that absorbs the supported input shapes.

To register a new wrapper, add it to ``APPROVED_INPUT_WRAPPERS`` (and add a
sample bare input for the behavioral coercion test).
"""

from __future__ import annotations

import pytest

from proto_tools.entities.structures import Structure

APPROVED_INPUT_WRAPPERS = [
    Structure,
]

# A minimally valid PDB string — single residue is fine for shape coercion;
# Structure._validate_structure (mode='after') rejects truly malformed input.
_MINIMAL_PDB = (
    "ATOM      1  N   MET A   1      11.104  13.207  10.300  1.00 20.00           N  \n"
    "ATOM      2  CA  MET A   1      11.804  14.247  11.040  1.00 20.00           C  \n"
    "ATOM      3  C   MET A   1      13.304  14.011  10.940  1.00 20.00           C  \n"
    "ATOM      4  O   MET A   1      13.804  13.001  10.440  1.00 20.00           O  \n"
    "END\n"
)

_SAMPLE_BARE_INPUTS: dict[type, list[object]] = {
    Structure: [_MINIMAL_PDB],
}


@pytest.mark.parametrize("wrapper", APPROVED_INPUT_WRAPPERS, ids=lambda w: w.__name__)
def test_wrapper_has_before_validator(wrapper: type) -> None:
    """Approved wrapper must define a ``model_validator(mode='before')``.

    Without a before-validator, Pydantic can't coerce bare inputs (path strings,
    raw content, etc.) into the wrapper, and tool authors typing fields as the
    wrapper would silently get failures when callers pass anything other than
    the canonical ``{"field": value}`` envelope.
    """
    decorators = wrapper.__pydantic_decorators__.model_validators
    has_before = any(d.info.mode == "before" for d in decorators.values())
    assert has_before, (
        f"{wrapper.__name__} is in APPROVED_INPUT_WRAPPERS but has no "
        f"model_validator(mode='before'). Either remove it from the list, or "
        f"add a coercion validator that absorbs the supported input shapes."
    )


@pytest.mark.parametrize("wrapper", APPROVED_INPUT_WRAPPERS, ids=lambda w: w.__name__)
def test_wrapper_coerces_bare_input(wrapper: type) -> None:
    """Approved wrapper must accept bare inputs (not just ``{"field": value}`` dicts).

    This is the behavioral counterpart to ``test_wrapper_has_before_validator``:
    even when a before-validator exists, it must actually coerce the canonical
    bare-input shapes.
    """
    samples = _SAMPLE_BARE_INPUTS.get(wrapper)
    assert samples, (
        f"{wrapper.__name__} is in APPROVED_INPUT_WRAPPERS but has no entry in "
        f"_SAMPLE_BARE_INPUTS. Add one valid bare input."
    )
    for bare in samples:
        obj = wrapper.model_validate(bare)
        assert isinstance(obj, wrapper), f"{wrapper.__name__} did not coerce {bare!r:.50} into {wrapper.__name__}"
