"""
RFdiffusion3 inference implementation.

References:
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/intro_inference_calculations.md
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import gemmi

logger = logging.getLogger(__name__)

class RFdiffusion3Model:
    """RFdiffusion3 model for protein structure design.

    References:
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    _STRUCTURE_EXTENSIONS = (".pdb", ".cif", ".pdb.gz", ".cif.gz")
    _AA_MAP = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    def __init__(self):
        """Initialize RFdiffusion3 model wrapper."""
        self._loaded = False
        self.rfd3_executable = None

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        # Common parameters (explicit for discoverability)
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        num_timesteps: int = 200,
        step_scale: float = 1.5,
        low_memory_mode: bool = False,
        ckpt_path: str = "rfd3",
        verbose: bool = False,
        # All other CLI args pass through to rfd3
        **cli_kwargs,
    ) -> Dict[str, Any]:
        """
        Run RFdiffusion3 structure design.

        This method supports all CLI arguments documented in the RFdiffusion3 input specification.
        Common parameters are explicit; all other RFdiffusion3 CLI args can be passed as keyword arguments.

        For the full list of available CLI arguments, see:
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

        Args:
            input_json_path: Path to input JSON file with design specifications
            output_dir: Directory to write output to
            n_batches: number of batches to generate per input key (default: 1).
            diffusion_batch_size: number of diffusion samples (designs) per batch (default: 8).
            num_timesteps: diffusion timesteps for sampling (default: 200).
            step_scale: scales diffusion step size; higher → less diverse, more designable (default: 1.5).
            low_memory_mode: memory-efficient tokenization mode; set True if GPU RAM is tight (default: False).
            ckpt_path: String containing the path and file name of the checkpoint path
                you want to use (default: rfd3).
            verbose: Whether to print status messages
            **cli_kwargs: Additional CLI arguments passed directly to rfd3.
                See RFdiffusion3 docs for complete list.

        Returns:
            Dictionary containing list of designed structures with metadata

        References:
            RFdiffusion3 Input Specification:
            https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
        """
        # Lazy load on first call
        if not self._loaded:
            self.load(verbose)

        # Build command: rfd3 design out_dir=<dir> inputs=<json> [options]
        cmd = [
            self.rfd3_executable,
            "design",
            f"out_dir={output_dir}",
            f"inputs={input_json_path}",
            f"n_batches={n_batches}",
            f"diffusion_batch_size={diffusion_batch_size}",
            f"inference_sampler.num_timesteps={num_timesteps}",
            f"inference_sampler.step_scale={step_scale}",
            f"low_memory_mode={low_memory_mode}",
            f"ckpt_path={ckpt_path}",
        ]

        # Add all additional CLI kwargs
        for key, value in cli_kwargs.items():
            # Handle specification dict - flatten to CLI format
            if key == "specification":
                for spec_key, spec_value in value.items():
                    if isinstance(spec_value, (dict, list)):
                        cmd.append(f"specification.{spec_key}={json.dumps(spec_value)}")
                    else:
                        cmd.append(f"specification.{spec_key}={spec_value}")
                continue

            if value is None:
                continue
            elif isinstance(value, (dict, list)):
                cmd.append(f"{key}={json.dumps(value)}")
            else:
                cmd.append(f"{key}={value}")

        logger.debug(f"Running RFdiffusion3 command: {' '.join(cmd)}")

        # Run the command
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            env=os.environ,
            encoding="utf-8",
        )

        if result.stdout:
            logger.debug(result.stdout)

        # Extract the outputs
        return self._extract_rfd3_outputs(output_dir)

    def load(self, verbose: bool = False):
        """Load RFdiffusion3 model components."""
        logger.debug("Initializing RFdiffusion3")

        # Try venv bin directory first, then PATH
        venv_rfdiffusion3 = Path(sys.executable).parent / "rfd3"
        self.rfd3_executable = str(venv_rfdiffusion3) if venv_rfdiffusion3.exists() else shutil.which("rfd3")
        if not self.rfd3_executable:
            raise ImportError("Could not find 'rfd3' executable. rc-foundry[rfd3] must be installed.")
        self._loaded = True

        logger.debug(f"RFdiffusion3 initialized. Executable: {self.rfd3_executable}")

    def _extract_rfd3_outputs(self, output_dir: str) -> Dict[str, Any]:
        """Extract designed structures from RFdiffusion3 outputs.

        RFdiffusion3 output files follow this naming convention:
            <input_json_name>_<spec_key>_<batch>_model_<n>.<ext> (e.g. rfdiffusion3_input_spec-0_0_model_0.cif.gz)

        Each structure file has a corresponding .json file with metadata

        See: https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/intro_inference_calculations.md
        """
        output_dir = Path(output_dir)
        structure_files = sorted(
            f for f in output_dir.iterdir()
            if f.name.endswith(self._STRUCTURE_EXTENSIONS)
        )

        designs = []
        for struct_file in structure_files:
            stem = self._strip_ext(struct_file.name)
            content = self._read_structure_file(struct_file)

            # Parse spec_key and design_index from filename
            model_match = re.search(r'_model_(\d+)$', stem)
            design_index = int(model_match.group(1)) if model_match else 0
            stem_no_model = stem[:model_match.start()] if model_match else stem

            spec_match = re.search(r'_(spec-\d+)_', stem_no_model)
            spec_key = spec_match.group(1) if spec_match else stem_no_model.rsplit('_', 2)[-2]

            designs.append({
                "structure_content": content,
                "sequence": self._extract_sequence(content, is_cif=".cif" in struct_file.name),
                "spec_key": spec_key,
                "design_index": design_index,
                "metadata": self._read_output_json(struct_file),
            })

        return {"designs": designs}

    def _extract_sequence(self, content: str, is_cif: bool = False) -> str:
        """Extract sequence from structure content (chains separated by /).

        Uses 'X' for unknown residues (ligands, modified AAs, nucleotides) to preserve
        index alignment with structure. Critical for all-atom design with RFdiffusion3.
        """
        if is_cif:
            doc = gemmi.cif.read_string(content)
            structure = gemmi.make_structure_from_block(doc[0])
        else:
            structure = gemmi.read_pdb_string(content)

        sequences = []
        for chain in structure[0]:
            seq = "".join(self._AA_MAP.get(res.name, "X") for res in chain)
            if seq:
                sequences.append(seq)
        return "/".join(sequences)

    def _strip_ext(self, filename: str) -> str:
        """Strip structure file extensions (.cif.gz, .pdb.gz, .cif, .pdb)."""
        for ext in self._STRUCTURE_EXTENSIONS:
            if filename.endswith(ext):
                return filename[:-len(ext)]
        return filename

    def _read_output_json(self, structure_file: Path) -> Dict[str, Any]:
        """Read the JSON metadata file corresponding to a structure file."""
        json_path = structure_file.parent / f"{self._strip_ext(structure_file.name)}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
        return {}

    def _read_structure_file(self, file_path: Path) -> str:
        """Read structure file content, handling gzip compression."""
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rt") as f:
                return f.read()
        return file_path.read_text()

# Standalone script entry point for venv execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    # Get the input and output json paths
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    # Read input json
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Extract required args, pass everything else as cli_kwargs
    rfdiffusion3_input_json = input_data.pop("input_json_path")
    rfdiffusion3_output_dir = input_data.pop("output_dir")

    # Create model and run inference
    model = RFdiffusion3Model()
    output_data = model(
        input_json_path=rfdiffusion3_input_json,
        output_dir=rfdiffusion3_output_dir,
        verbose=True,
        **input_data,  # All other args pass through
    )

    # Write the output to a json file
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
