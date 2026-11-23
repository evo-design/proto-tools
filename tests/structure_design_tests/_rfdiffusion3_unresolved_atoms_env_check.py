"""In-env check that rfd3's own loader reads a staged input with no NaN coordinates.

``argv[1]`` is the original input and ``argv[2]`` the file ``to_json_spec`` staged from it.
The original is loaded first as the control: rfd3's ``inference_load_`` must keep its
unresolved atoms as NaN coordinates, or the staging step would no longer be what stands
between them and the denoiser. The staged file must load with no NaN coordinates and
exactly the original's resolved atoms, coordinates and annotations.

Run by ``test_rfdiffusion3_staged_input_loads_without_nan_in_rfd3`` in ``test_rfdiffusion3.py``.
Plain asserts; a non-zero exit surfaces as ``RuntimeError``.
"""

import sys

import numpy as np
from rfd3.utils.inference import inference_load_


def main() -> int:
    """Load both files with rfd3's inference loader and compare their atoms."""
    original = inference_load_(sys.argv[1])["atom_array"]
    staged = inference_load_(sys.argv[2])["atom_array"]

    unresolved = np.isnan(original.coord).any(axis=-1)
    assert unresolved.any(), "rfd3 no longer keeps NaN atoms from the input; the control did not run"
    print(f"CONTROL original keeps {int(unresolved.sum())} NaN-coordinate atoms")

    assert not np.isnan(staged.coord).any(), "staged input still has NaN coordinates"
    resolved = original[~unresolved]
    assert staged.array_length() == resolved.array_length(), (staged.array_length(), resolved.array_length())
    np.testing.assert_array_equal(staged.coord, resolved.coord)
    for name in ("chain_id", "res_id", "res_name", "atom_name", "b_factor", "occupancy"):
        np.testing.assert_array_equal(staged.get_annotation(name), resolved.get_annotation(name), err_msg=name)
    print(f"PASS staged input has {staged.array_length()} atoms, none NaN, identical to the resolved atoms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
