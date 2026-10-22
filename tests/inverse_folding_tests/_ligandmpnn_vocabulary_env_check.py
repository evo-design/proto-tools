"""In-env check that LigandMPNN's native token orders map onto proto's amino-acid columns.

Both scoring paths resolve their vocabulary at call time — the Foundry engine from
``MPNN_TOKEN_ENCODING``, the legacy script from ``data_utils.restype_int_to_str`` — so neither
order is a constant this repo can pin. What the worker relies on is that
``gather_indices(AMINO_ACIDS_LIST, native)`` maps whatever it finds onto proto's 20 columns.
This drives that mapping with distinguishable values, so a transposition shows up as a wrong
residue. Neither import needs model weights. The Foundry encoding is always available; the
legacy script's source is provisioned with its checkpoint, so it is checked when present and
reported as skipped when not.

Run by ``test_ligandmpnn_native_vocabularies_map_to_proto_columns`` in ``test_ligandmpnn.py``.
Plain asserts; a non-zero exit surfaces as ``RuntimeError``.
"""

from __future__ import annotations

import numpy as np
from standalone_helpers.vocabulary import AMINO_ACIDS_LIST, gather_indices


def _check_mapping(label: str, native: list[str]) -> None:
    """Assert the native order gathers onto proto's columns, value by value."""
    assert len(native) >= len(AMINO_ACIDS_LIST), f"{label}: only {len(native)} tokens"

    columns = gather_indices(AMINO_ACIDS_LIST, native, label=label)
    # Column j carries the value j, so a column's value names the native column it came from.
    matrix = np.tile(np.arange(len(native), dtype=float), (2, 1))

    proto_ordered = matrix[:, columns]

    assert proto_ordered.shape == (2, len(AMINO_ACIDS_LIST)), f"{label}: {proto_ordered.shape}"
    for column, residue in enumerate(AMINO_ACIDS_LIST):
        expected = native.index(residue)
        assert proto_ordered[0, column] == expected, (
            f"{label}: proto column {column} ({residue}) carries native column "
            f"{proto_ordered[0, column]:.0f}, expected {expected}"
        )
    extra = [token for token in native if token not in AMINO_ACIDS_LIST]
    for token in extra:
        assert float(native.index(token)) not in set(proto_ordered[0]), f"{label}: {token!r} column reached the output"
    print(f"PASS {label}: {len(native)} native tokens -> 20 proto columns, dropping {extra}")


def main() -> int:
    """Check both scoring paths' vocabularies, reporting each."""
    from atomworks.constants import DICT_THREE_TO_ONE
    from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING

    foundry = [DICT_THREE_TO_ONE[MPNN_TOKEN_ENCODING.idx_to_token[i]] for i in range(MPNN_TOKEN_ENCODING.n_tokens)]
    _check_mapping("foundry", foundry)

    import inference

    # The legacy script's source is provisioned alongside its checkpoint, so it is only present
    # on a machine that has run that path. Report which paths were covered either way.
    package_path = inference._legacy_package_path(None)
    if package_path is None:
        print("SKIP legacy: source not provisioned on this machine (it ships with the checkpoint)")
    else:
        data_utils = inference._load_legacy_modules(package_path)["data_utils"]
        legacy = [data_utils.restype_int_to_str[i] for i in range(len(data_utils.restype_int_to_str))]
        _check_mapping("legacy", legacy)

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
