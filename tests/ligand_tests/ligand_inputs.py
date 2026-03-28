"""
tests/ligand_tests/ligand_inputs.py

Contains input data for ligand tests.
"""

from pathlib import Path

_prefix_path = Path(__file__).parent.parent / "dummy_data" / "ligand_input_test_examples"

SINGLE_FRAGMENT_SMI = _prefix_path / "PubChem5284596.smi"
SINGLE_FRAGMENT_2D_SDF = _prefix_path / "PubChem5284596_2D.sdf"
SINGLE_FRAGMENT_3D_SDF = _prefix_path / "PubChem5284596_3D.sdf"

MULTIPLE_FRAGMENT_SMI = _prefix_path / "PubChem11597697.smi"
MULTIPLE_FRAGMENT_2D_SDF = _prefix_path / "PubChem11597697_2D.sdf"
MULTIPLE_FRAGMENT_3D_SDF = _prefix_path / "PubChem11597697_3D.sdf"

LIGAND_TEST_FILES = {
    "single_fragment": {
        "smi": SINGLE_FRAGMENT_SMI,
        "2d_sdf": SINGLE_FRAGMENT_2D_SDF,
        "3d_sdf": SINGLE_FRAGMENT_3D_SDF,
    },
    "multiple_fragment": {
        "smi": MULTIPLE_FRAGMENT_SMI,
        "2d_sdf": MULTIPLE_FRAGMENT_2D_SDF,
        "3d_sdf": MULTIPLE_FRAGMENT_3D_SDF,
    },
}
