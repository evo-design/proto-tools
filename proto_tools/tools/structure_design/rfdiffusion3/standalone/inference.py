"""RFdiffusion3 inference implementation.

References:
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/intro_inference_calculations.md
"""

import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import gemmi
from standalone_helpers import set_torch_seed

logger = logging.getLogger(__name__)


class RFdiffusion3Model:
    """RFdiffusion3 model for protein structure design.

    References:
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    _STRUCTURE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".pdb", ".cif", ".pdb.gz", ".cif.gz")
    _AA_MAP: ClassVar[dict[str, str]] = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    def __init__(self) -> None:
        """Initialize RFdiffusion3 model wrapper."""
        self._loaded = False
        self.rfd3_executable: str | None = None

    def __call__(
        self,
        input_json_path: str,
        output_dir: str,
        # Device for subprocess environment
        device: str = "cuda",
        # Common parameters (explicit for discoverability)
        n_batches: int = 1,
        diffusion_batch_size: int = 8,
        num_timesteps: int = 200,
        step_scale: float = 1.5,
        low_memory_mode: bool = False,
        ckpt_path: str = "rfd3",
        seed: int | None = None,
        verbose: bool = False,
        # All other CLI args pass through to rfd3
        **cli_kwargs: Any,
    ) -> dict[str, Any]:
        """Run RFdiffusion3 structure design.

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
            seed: Random seed for reproducibility (default: None).
            verbose: Whether to print status messages
            **cli_kwargs: Additional CLI arguments passed directly to rfd3.
                See RFdiffusion3 docs for complete list.

            device: Target device for model execution.

        Returns:
            Dictionary containing list of designed structures with metadata

        References:
            RFdiffusion3 Input Specification:
            https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
        """
        # Seed Python-side RNG. The rfd3 subprocess also receives seed as
        # a CLI argument, but does not yet produce deterministic results
        # due to upstream non-determinism (RosettaCommons/foundry#170).
        set_torch_seed(seed)

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

        # Add seed if provided (passed to rfd3's RNG for reproducibility)
        if seed is not None:
            cmd.append(f"seed={seed}")

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
            if isinstance(value, (dict, list)):
                cmd.append(f"{key}={json.dumps(value)}")
            else:
                cmd.append(f"{key}={value}")

        logger.debug(f"Running RFdiffusion3 command: {' '.join(cmd)}")  # type: ignore[arg-type]

        # Get subprocess environment with correct CUDA_VISIBLE_DEVICES
        from standalone_helpers import get_subprocess_device_env

        env = get_subprocess_device_env(device)

        # Run the command
        result = subprocess.run(
            cmd,  # type: ignore[arg-type]
            check=True,
            text=True,
            env=env,
            encoding="utf-8",
        )

        if result.stdout:
            logger.debug(result.stdout)

        # Extract the outputs
        return self._extract_rfd3_outputs(output_dir)

    def load(self, verbose: bool = False) -> None:  # noqa: ARG002 — required by tool interface
        """Load RFdiffusion3 model components."""
        logger.debug("Initializing RFdiffusion3")

        # Set FOUNDRY_CHECKPOINT_DIRS so Foundry finds BPT-managed weights
        from standalone_helpers import resolve_weights_dir

        weights_dir = resolve_weights_dir("rfdiffusion3")
        if weights_dir:
            os.environ["FOUNDRY_CHECKPOINT_DIRS"] = weights_dir

        # Try venv bin directory first, then PATH
        venv_rfdiffusion3 = Path(sys.executable).parent / "rfd3"
        exe = str(venv_rfdiffusion3) if venv_rfdiffusion3.exists() else shutil.which("rfd3")
        if not exe:
            raise ImportError("Could not find 'rfd3' executable. rc-foundry[rfd3] must be installed.")
        self.rfd3_executable = exe
        self._loaded = True

        logger.debug(f"RFdiffusion3 initialized. Executable: {self.rfd3_executable}")

    def _extract_rfd3_outputs(self, output_dir: str) -> dict[str, Any]:
        """Extract designed structures from RFdiffusion3 outputs.

        RFdiffusion3 output files follow this naming convention:
            <input_json_name>_<spec_key>_<batch>_model_<n>.<ext> (e.g. rfdiffusion3_input_spec-0_0_model_0.cif.gz)

        Each structure file has a corresponding .json file with metadata

        See: https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/intro_inference_calculations.md
        """
        output_dir = Path(output_dir)  # type: ignore[assignment]
        structure_files = sorted(
            f
            for f in output_dir.iterdir()  # type: ignore[attr-defined]
            if f.name.endswith(self._STRUCTURE_EXTENSIONS)
        )

        designs = []
        for struct_file in structure_files:
            stem = self._strip_ext(struct_file.name)
            content = self._read_structure_file(struct_file)

            # Parse spec_key and design_index from filename
            model_match = re.search(r"_model_(\d+)$", stem)
            design_index = int(model_match.group(1)) if model_match else 0
            stem_no_model = stem[: model_match.start()] if model_match else stem

            spec_match = re.search(r"_(spec-\d+)_", stem_no_model)
            spec_key = spec_match.group(1) if spec_match else stem_no_model.rsplit("_", 2)[-2]

            designs.append(
                {
                    "structure_content": content,
                    "sequence": self._extract_sequence(content, is_cif=".cif" in struct_file.name),
                    "spec_key": spec_key,
                    "design_index": design_index,
                    "metadata": self._read_output_json(struct_file),
                }
            )

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
                return filename[: -len(ext)]
        return filename

    def _read_output_json(self, structure_file: Path) -> dict[str, Any]:
        """Read the JSON metadata file corresponding to a structure file."""
        json_path = structure_file.parent / f"{self._strip_ext(structure_file.name)}.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {}

    def _read_structure_file(self, file_path: Path) -> str:
        """Read structure file content, handling gzip compression."""
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rt") as f:
                return f.read()
        return file_path.read_text()


# ============================================================================
# Dispatch
# ============================================================================
_model: RFdiffusion3Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = RFdiffusion3Model()

    kwargs = dict(input_dict)
    kwargs.pop("operation", None)
    device = kwargs.pop("device", "cuda")  # Extract device for subprocess environment
    rfdiffusion3_input_json = kwargs.pop("input_json_path")
    rfdiffusion3_output_dir = kwargs.pop("output_dir")
    seed = kwargs.pop("seed", None)

    return _model(
        input_json_path=rfdiffusion3_input_json,
        output_dir=rfdiffusion3_output_dir,
        device=device,
        seed=seed,
        verbose=kwargs.pop("verbose", False),
        **kwargs,
    )


def to_device(device: str) -> dict[str, Any]:
    """Passthrough for CLI tool - automatically unloads after each call."""
    # CLI tool that spawns subprocesses and naturally unloads after each call
    # This is a passthrough for standardization with other tools
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(device=0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
