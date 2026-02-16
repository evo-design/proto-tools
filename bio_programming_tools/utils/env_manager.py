"""
Environment utilities and the EnvManager class to create and manage isolated
venvs for models with difficult dependencies.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .device import determine_visible_devices

logger = getLogger(__name__)

def seed_everything(seed: int):
    """
    Seeds everything

    Args:
        seed (int): The seed to use
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class EnvManager:
    """
    Class that manages the creation and management of venvs for different models.
    """

    def __init__(self, model_name: str, refresh: bool = False):
        """
        Initialize a EnvManager for a given model.

        Args:
            model_name: The name of the model to manage
            refresh: Whether to refresh the venv if it already exists
        """
        self.model_name = self._determine_valid_model_name(model_name)
        venv_root = self._get_venvs_root()
        self.env_path = venv_root / f"{model_name}_env"
        self.setup_script = self._find_setup_script(model_name)

        # auto-create/refresh venv if needed
        if (
            not self.env_path.exists()
            or refresh
            or not self._is_venv_setup_successful()
        ):
            if self.env_path.exists() and not self._is_venv_setup_successful():
                logger.info(
                    f"Venv for {model_name} exists but setup was not successful. Attempting to recreate..."
                )
            else:
                logger.info(f"Setting up venv for {model_name}...")
            self._create_env()

        else:
            logger.debug(
                f"Venv for {model_name} already exists and setup was successful at {self.env_path}"
            )

    @staticmethod
    def _get_venvs_root() -> Path:
        """
        Determine the .venvs root directory.

        For editable installs (pip install -e .), finds the project root by
        walking up from this file looking for pyproject.toml, then uses
        project_root/.venvs/.

        For non-editable installs (pip install .), the package is copied into
        site-packages and there's no project root. Falls back to a user-level
        cache directory: $XDG_CACHE_HOME/bio_programming_tools/.venvs/ or
        ~/.cache/bio_programming_tools/.venvs/.
        """
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                venvs = parent / ".venvs"
                venvs.mkdir(parents=True, exist_ok=True)
                return venvs
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        venvs = cache_home / "bio_programming_tools" / ".venvs"
        venvs.mkdir(parents=True, exist_ok=True)
        return venvs

    def _determine_valid_model_name(self, model_name: str):
        """
        Helper function to determine if a provided model is a model that contains
        a 'standalone' subdirectory.
        """
        # Get tools directory (utils/ -> bio_programming_tools/ -> tools/)
        tools_dir = Path(__file__).parent.parent / "tools"
        available_models = []

        # Find all directories that contain a "standalone" subdirectory
        for item in tools_dir.rglob("*"):
            if item.is_dir() and (item / "standalone").exists():
                # Use just the final directory name
                available_models.append(item.name)

        if model_name not in available_models:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models: {available_models}"
            )
        return model_name

    def _find_setup_script(self, model_name: str):
        """
        Helper function to find the setup.sh script for a given model.

        Searches recursively for directories matching model_name that contain
        a standalone subdirectory with a valid setup.sh file. If multiple
        directories match, prefers the one with a valid setup.sh and logs a warning.

        Args:
            model_name: The name of the model to find setup.sh for

        Returns:
            Path to the setup.sh script

        Raises:
            ValueError: If no valid setup.sh is found for the model
        """
        # Get tools directory (utils/ -> bio_programming_tools/ -> tools/)
        tools_dir = Path(__file__).parent.parent / "tools"

        # Find all directories that match the model name and have a standalone subdirectory
        matching_dirs = []
        for item in tools_dir.rglob("*"):
            if item.is_dir() and item.name == model_name and (item / "standalone").exists():
                matching_dirs.append(item)

        if not matching_dirs:
            raise ValueError(
                f"Could not find any standalone directory for model '{model_name}' "
                f"in {tools_dir}"
            )

        # Find directories with valid setup.sh files
        valid_dirs = []
        for dir_path in matching_dirs:
            setup_script = dir_path / "standalone" / "setup.sh"
            if setup_script.exists() and setup_script.is_file():
                valid_dirs.append((dir_path, setup_script))

        if not valid_dirs:
            # Found matching directories but none have valid setup.sh
            dirs_list = "\n  ".join(str(d) for d in matching_dirs)
            raise ValueError(
                f"Found {len(matching_dirs)} standalone director{'y' if len(matching_dirs) == 1 else 'ies'} "
                f"for model '{model_name}', but none contain a valid setup.sh file:\n  {dirs_list}"
            )

        if len(valid_dirs) > 1:
            # Multiple valid directories found - warn and use the first one
            dirs_list = "\n  ".join(str(d) for d, _ in valid_dirs)
            logger.warning(
                f"Found {len(valid_dirs)} valid standalone directories for model '{model_name}':\n  "
                f"{dirs_list}\n"
                f"Using: {valid_dirs[0][0]}\n"
                f"This may indicate stale directories from a refactor. Consider cleaning up duplicate directories."
            )

        return valid_dirs[0][1]

    def _is_venv_setup_successful(self):
        """
        Helper function to check if the venv setup was successful by reading STATUS.txt
        and verifying the Python executable exists and works
        """
        status_file = self.env_path / "STATUS.txt"
        if not status_file.exists():
            return False

        try:
            with open(status_file, "r") as f:
                status = f.read().strip()
            if status != "SUCCESS":
                return False

            # Also verify the Python executable exists and is functional
            python_exe = self.env_path / "bin" / "python"
            if not python_exe.exists():
                logger.warning(
                    f"STATUS.txt says SUCCESS but Python executable missing at {python_exe}"
                )
                return False

            # Test that the Python executable actually works (e.g., symlink isn't broken)
            try:
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    timeout=30,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning(
                        f"Python executable at {python_exe} exists but doesn't work (exit code {result.returncode})"
                    )
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                logger.warning(
                    f"Python executable at {python_exe} is not functional: {e}"
                )
                return False

            return True
        except Exception:
            return False

    def _get_clean_subprocess_env(self) -> Dict[str, str]:
        """
        Create a clean environment for subprocesses by removing Jupyter/IPython
        specific variables and CUDA library paths that can cause issues.

        Jupyter notebooks set environment variables that are only valid within
        the Jupyter context. When these are inherited by subprocesses, they can
        cause errors. For example, MPLBACKEND='module://matplotlib_inline.backend_inline'
        causes matplotlib to fail in non-Jupyter contexts.

        Additionally, LD_LIBRARY_PATH pointing to system CUDA libraries can
        interfere with bundled CUDA libraries in packages like JAX and PyTorch,
        causing them to fail to detect GPUs properly.

        Returns:
            A copy of os.environ with problematic variables removed.
        """
        JUPYTER_BLOCKLIST = {
            'MPLBACKEND',           # Jupyter matplotlib backend - causes immediate error
            'DISPLAY',              # X11 display, invalid in some contexts
            'JPY_PARENT_PID',       # Jupyter process identifiers
            'JPY_SESSION_NAME',
            'PYDEVD_USE_CYTHON',    # PyDev debugger flags
            'PYDEVD_USE_FRAME_EVAL',
            'JUPYTER_PLATFORM_DIRS', # Jupyter paths
            'JUPYTER_DATA_DIR',
            'JUPYTER_CONFIG_DIR',
            'JUPYTER_RUNTIME_DIR',
            'JUPYTER_PATH',
        }

        CUDA_BLOCKLIST = {
            'LD_LIBRARY_PATH',      # Can override bundled CUDA libraries in venv
        }

        CONDA_BLOCKLIST = {
            'CONDA_PREFIX',         # Points to conda env; can confuse pip/uv
            'CONDA_DEFAULT_ENV',    # Active conda env name
            'CONDA_PYTHON_EXE',    # Conda's Python, not the venv's
            'CONDA_PROMPT_MODIFIER',
            'CONDA_SHLVL',
            'CONDA_EXE',
            '_CE_CONDA',
            '_CONDA_EXE',
            '_CONDA_ROOT',
        }

        env = os.environ.copy()
        for key in JUPYTER_BLOCKLIST | CUDA_BLOCKLIST | CONDA_BLOCKLIST:
            env.pop(key, None)
        return env

    def _create_env(self):
        """
        Helper function to create a venv for a given model using the setup.sh script.
        """
        import datetime

        status_file = self.env_path / "STATUS.txt"

        try:
            # If venv exists but is broken, remove it first
            if self.env_path.exists():
                if not self._is_venv_setup_successful():
                    logger.info(f"Removing broken venv at {self.env_path}")
                    shutil.rmtree(self.env_path)

            # create venv with --copies to make it independent of base Python
            subprocess.run(
                [sys.executable, "-m", "venv", "--copies", str(self.env_path)], check=True
            )

            # Check if setup script exists
            if not self.setup_script.exists():
                error_msg = f"No setup.sh script found for {self.model_name} at {self.setup_script}"
                logger.error(f"EnvManager: {error_msg}")
                with open(status_file, "w") as f:
                    f.write(
                        f"FAILED\n\nError: {error_msg}\nTimestamp: {datetime.datetime.now()}\n"
                    )
                raise ValueError(error_msg)

            # Make setup script executable
            subprocess.run(["chmod", "+x", str(self.setup_script)], check=True)

            # Set up environment variables for the setup script
            env = self._get_clean_subprocess_env()
            env["VENV_PATH"] = str(self.env_path.absolute())
            env["PYTHON_EXE"] = str(self.env_path.absolute() / "bin" / "python")
            env["PIP_EXE"] = str(self.env_path.absolute() / "bin" / "pip")

            # Run the setup script from its directory with venv activated.
            # Stream stdout live so users see progress (e.g. download bars),
            # while capturing both streams for error reporting.
            activate_script = self.env_path.absolute() / "bin" / "activate"
            proc = subprocess.Popen(
                ["bash", "-c", f"source {activate_script} && {self.setup_script}"],
                cwd=self.setup_script.parent,
                env=env,
                stdout=None,  # stream to user's terminal/notebook
                stderr=subprocess.PIPE,
                text=True,
            )
            _, stderr_output = proc.communicate()

            # Create STATUS.txt based on result
            if proc.returncode == 0:
                # Success
                logger.debug(f"EnvManager: venv setup completed for {self.model_name}")
                with open(status_file, "w") as f:
                    f.write("SUCCESS")
            else:
                # Failure
                logger.error(
                    f"EnvManager: venv setup failed for {self.model_name} (exit code {proc.returncode})"
                )
                if stderr_output:
                    logger.error(f"EnvManager: stderr: {stderr_output}")
                with open(status_file, "w") as f:
                    f.write("FAILED\n\n")
                    f.write(f"Return code: {proc.returncode}\n")
                    f.write(f"Command: {self.setup_script}\n")
                    f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
                    if stderr_output:
                        f.write(f"STDERR:\n{stderr_output}\n")

                status_content = status_file.read_text()
                logger.error(
                    f"Setup failed for {self.model_name}.\n{status_content}"
                )

                # Provide helpful hints based on common error patterns
                hints = []
                if stderr_output:
                    if "command not found" in stderr_output:
                        hints.append("Hint: A required command was not found. Check that all dependencies are installed.")
                    if "micromamba" in stderr_output or "mamba" in stderr_output:
                        hints.append("Hint: Issue with micromamba installation. Check network connectivity and try clearing the venv.")
                    if "curl" in stderr_output or "download" in stderr_output.lower():
                        hints.append("Hint: Network download issue. Verify internet connectivity and firewall settings.")
                    if "permission denied" in stderr_output.lower():
                        hints.append("Hint: Permission issue. Check file/directory permissions.")

                if hints:
                    hint_text = "\n".join(hints)
                    logger.error(f"\n{hint_text}")

                raise subprocess.CalledProcessError(
                    proc.returncode,
                    str(self.setup_script),
                    None,
                    stderr_output,
                )

        except Exception as e:
            # Handle any other exceptions (like venv creation failure)
            logger.error(f"EnvManager: venv creation failed for {self.model_name}: {e}")
            if not status_file.exists():
                with open(status_file, "w") as f:
                    f.write("FAILED\n\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {datetime.datetime.now()}\n")
            raise

    def call_standalone_script_in_venv(
        self,
        script_path: Path,
        input_dict: Dict[str, Any],
        device: str = "cuda",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Helper function to call a standalone script utilizing the current venv

        Args:
            script_path: The path to the script to call with the venv activated
            input_dict: A dictionary of json-serializable input parameters to
                pass to the script
            device: The device to utilize for script execution for the model
            verbose: Whether to print verbose output from the script

        Returns:
            The output of the script as a dictionary
        """

        with tempfile.TemporaryDirectory() as temp_dir:

            # Create a temp input file location
            input_json_path = Path(temp_dir) / "input.json"
            with open(input_json_path, "w") as f:
                json.dump(input_dict, f)

            # Create a temp output file location
            output_json_path = Path(temp_dir) / "output.json"

            # Set up environment variables
            env = self._get_clean_subprocess_env()

            # Set CUDA_VISIBLE_DEVICES to the specified device number
            env["CUDA_VISIBLE_DEVICES"] = determine_visible_devices(device=device)

            try:
                if verbose:
                    logger.info(
                        f"Running {script_path} with input: {input_dict} and device: {device}"
                    )
                subprocess.run(
                    [
                        str(self.env_path.absolute() / "bin" / "python"),
                        str(script_path),
                        str(input_json_path),
                        str(output_json_path),
                    ],
                    env=env,
                    text=True,
                    check=True,
                    stdout=None if verbose else subprocess.PIPE,
                    stderr=None if verbose else subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running {script_path}: {e}")
                if e.stderr:
                    logger.error(f"STDERR: {e.stderr}")
                if e.stdout:
                    logger.error(f"STDOUT: {e.stdout}")
                raise e

            # Read in the output file
            with open(output_json_path, "r") as f:
                output_data = json.load(f)

            return output_data
