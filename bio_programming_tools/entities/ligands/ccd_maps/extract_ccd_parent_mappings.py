#!/usr/bin/env python3
"""
Extract parent residue/base mappings from wwPDB Chemical Component Dictionary.

This script reads the components.cif file and extracts mappings from modified
residues and bases (CCD codes) to their parent (canonical) amino acids or nucleotides.

Covers:
- Protein PTMs (post-translational modifications): SEP, TPO, PTR, MSE, etc.
- RNA modifications: 2MG, 5MC, PSU, etc.
- DNA modifications: 6MA, 8OG, etc.

Download the source file from: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz

Input: components.cif (or components.cif.gz)
Output: ccd_to_og.csv

Format of output CSV:
    ccd_code,parent_3letter,parent_1letter

Example entries:
    SEP,SER,S    # Phosphoserine -> Serine (protein PTM)
    TPO,THR,T    # Phosphothreonine -> Threonine (protein PTM)
    2MG,G,G      # 2'-O-methylguanosine -> Guanosine (RNA modification)
    6MA,DA,A     # N6-methyladenine -> Adenine (DNA modification)
"""

import gzip
from pathlib import Path


def extract_parent_mappings(cif_path: Path, output_path: Path):
    """Extract parent residue mappings from CCD file.

    Args:
        cif_path: Path to components.cif or components.cif.gz
        output_path: Path to output CSV file
    """
    # Open file (handle both gzipped and uncompressed)
    if cif_path.suffix == '.gz':
        f = gzip.open(cif_path, 'rt')
    else:
        f = open(cif_path, 'r')

    try:
        # Parse CIF format
        current_id = None
        current_parent = None
        current_one_letter = None
        results = []

        for line in f:
            line = line.strip()

            # New component entry
            if line.startswith("data_"):
                # Save previous entry if it has a parent and is a single residue (not a peptide)
                if (current_id and
                    current_parent and current_parent != "?" and
                    current_one_letter and current_one_letter != "?" and
                    len(current_one_letter) == 1):  # Only single-letter codes (not peptides)
                    results.append((current_id, current_parent, current_one_letter))

                # Start new entry
                current_id = line[5:]  # Remove "data_" prefix
                current_parent = None
                current_one_letter = None

            # Extract parent component ID
            elif "_chem_comp.mon_nstd_parent_comp_id" in line:
                parts = line.split()
                if len(parts) >= 2:
                    current_parent = parts[-1]  # Last field is the value

            # Extract one-letter code
            elif "_chem_comp.one_letter_code" in line:
                parts = line.split()
                if len(parts) >= 2:
                    current_one_letter = parts[-1]  # Last field is the value

        # Save last entry if it has a parent and is a single residue (not a peptide)
        if (current_id and
            current_parent and current_parent != "?" and
            current_one_letter and current_one_letter != "?" and
            len(current_one_letter) == 1):  # Only single-letter codes (not peptides)
            results.append((current_id, current_parent, current_one_letter))

    finally:
        f.close()

    # Write results to CSV
    with open(output_path, 'w') as out:
        # Write header
        out.write("# CCD Code to Parent Residue/Base Mapping\n")
        out.write("# Extracted from wwPDB Chemical Component Dictionary\n")
        out.write("# Source: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif\n")
        out.write("#\n")
        out.write("# Covers protein PTMs, RNA modifications, and DNA modifications\n")
        out.write("#\n")
        out.write("# Format: CCD_CODE,PARENT_3LETTER,PARENT_1LETTER\n")
        out.write("#\n")
        out.write("# CCD_CODE: Three-letter code for modified residue or base\n")
        out.write("# PARENT_3LETTER: Three-letter code for parent (canonical) component\n")
        out.write("# PARENT_1LETTER: Single-letter code for parent (canonical) amino acid or nucleotide\n")
        out.write("#\n")
        out.write("ccd_code,parent_3letter,parent_1letter\n")

        # Write data sorted by CCD code
        for ccd_code, parent_3letter, one_letter in sorted(results):
            out.write(f"{ccd_code},{parent_3letter},{one_letter}\n")

    return len(results)


def main():
    """Main entry point."""
    # Find input file (try both .cif and .cif.gz)
    script_dir = Path(__file__).parent

    cif_path = script_dir / "components.cif"
    if not cif_path.exists():
        cif_path = script_dir / "components.cif.gz"
        if not cif_path.exists():
            print("ERROR: Could not find components.cif or components.cif.gz")
            print(f"       in directory: {script_dir}")
            return 1

    # Output path
    output_path = script_dir / "ccd_to_og.csv"

    print(f"Reading CCD file: {cif_path.name}")
    print(f"This may take a minute...")

    # Extract mappings
    num_mappings = extract_parent_mappings(cif_path, output_path)

    print(f"✓ Extracted {num_mappings} modified residue mappings")
    print(f"✓ Wrote output to: {output_path.name}")

    # Show some examples
    print("\nExample mappings:")
    with open(output_path, 'r') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip() and l != 'ccd_code,parent_3letter,parent_1letter\n']
        for line in lines[:10]:
            print(f"  {line.strip()}")

    return 0


if __name__ == "__main__":
    exit(main())
