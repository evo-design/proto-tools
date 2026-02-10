from pathlib import Path

from ..structure import Structure

GFP_CIF_PATH = Path(Path(__file__).parent / "gfp.cif").absolute()

def get_gfp_structure() -> Structure:
    return Structure(structure_filepath_or_content=GFP_CIF_PATH)
