"""
Tests for EnvManager class in bio_programming_tools.utils.env_manager

These tests verify that venvs can be created successfully for all models
with standalone directories. The venv creation tests are marked with @skip_ci
to avoid running in CI environments due to their resource-intensive nature.
"""
import pytest

from bio_programming_tools.utils.env_manager import EnvManager


# Models that require a GPU
GPU_MODELS = [
    "alphagenome",
    "bioemu",
    "boltz2",
    "borzoi",
    "chai1",
    "enformer",
    "esm2",
    "esm3",
    "esmfold",
    "evo2",
    "ligandmpnn",
    "progen2",
    "proteinmpnn",
    "protenix",
    "rfdiffusion3",
    "splice_transformer",
]

# Models that run on CPU only
CPU_MODELS = [
    "blast",
    "colabfold_search",
    "mafft",
    "mmseqs",
    "orfipy",
    "prodigal",
    "pyhmmer",
    "viennarna",
]

# All models with standalone directories
MODELS_WITH_STANDALONE = GPU_MODELS + CPU_MODELS


class TestEnvManagerValidation:
    """Lightweight tests for EnvManager validation logic that run in CI."""

    @pytest.mark.parametrize("model_name", MODELS_WITH_STANDALONE)
    def test_all_models_have_valid_names(self, model_name):
        """
        Test that all models with standalone directories are recognized as valid.

        This is a lightweight test that only validates model names without
        creating venvs, suitable for running in CI.
        """
        # This should not raise ValueError
        env_manager = EnvManager.__new__(EnvManager)
        env_manager.model_name = model_name

        # Verify the model name passes validation
        validated_name = env_manager._determine_valid_model_name(model_name)
        assert validated_name == model_name

    @pytest.mark.parametrize("model_name", MODELS_WITH_STANDALONE)
    def test_all_models_have_setup_scripts(self, model_name):
        """
        Test that all models have valid setup.sh scripts in their standalone directories.

        This verifies that the setup script can be found without actually executing it.
        """
        env_manager = EnvManager.__new__(EnvManager)
        env_manager.model_name = model_name

        # Find the setup script
        setup_script = env_manager._find_setup_script(model_name)

        # Verify the script exists and is a file
        assert setup_script.exists(), f"Setup script not found at {setup_script}"
        assert setup_script.is_file(), f"Setup script is not a file: {setup_script}"
        assert setup_script.name == "setup.sh", f"Unexpected setup script name: {setup_script.name}"


@pytest.mark.skip_ci
class TestEnvManagerVenvCreation:
    """
    Integration tests for EnvManager venv creation.

    These tests actually create venvs for each model, which is resource-intensive
    and time-consuming (potentially 30+ minutes per model). They are skipped in CI.

    Note: Created venvs are kept in .venvs/ directory after tests complete.
    """

    def _verify_venv(self, model_name: str) -> None:
        """Verify that a venv can be created and is functional for a model."""
        import subprocess

        # Create/verify the venv (refresh=False means reuse if exists)
        env_manager = EnvManager(model_name=model_name, refresh=False)

        # Verify venv directory exists
        assert env_manager.env_path.exists(), (
            f"Venv directory not created at {env_manager.env_path}"
        )
        assert env_manager.env_path.is_dir(), (
            f"Venv path exists but is not a directory: {env_manager.env_path}"
        )

        # Verify STATUS.txt indicates success
        status_file = env_manager.env_path / "STATUS.txt"
        assert status_file.exists(), (
            f"STATUS.txt not found at {status_file}"
        )

        with open(status_file, "r") as f:
            status_content = f.read().strip()

        assert status_content == "SUCCESS", (
            f"Venv setup failed for {model_name}. STATUS.txt content:\n{status_content}"
        )

        # Verify Python executable exists
        python_exe = env_manager.env_path / "bin" / "python"
        assert python_exe.exists(), (
            f"Python executable not found at {python_exe}"
        )

        # Verify Python executable is functional
        result = subprocess.run(
            [str(python_exe), "--version"],
            capture_output=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, (
            f"Python executable at {python_exe} is not functional. "
            f"Exit code: {result.returncode}, stderr: {result.stderr.decode()}"
        )

        # Verify setup script was found
        assert env_manager.setup_script.exists(), (
            f"Setup script not found at {env_manager.setup_script}"
        )

        # Log success for visibility during test runs
        print(f"\n✓ Successfully verified venv for {model_name} at {env_manager.env_path}")

    @pytest.mark.uses_gpu
    @pytest.mark.parametrize("model_name", GPU_MODELS)
    def test_create_venv_for_gpu_model(self, model_name):
        """Test venv creation for GPU models."""
        self._verify_venv(model_name)

    @pytest.mark.parametrize("model_name", CPU_MODELS)
    def test_create_venv_for_cpu_model(self, model_name):
        """Test venv creation for CPU models."""
        self._verify_venv(model_name)
