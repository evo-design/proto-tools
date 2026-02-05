"""
test_model_caching.py

Tests for model instance caching functionality across Evo2, ESM2, and ESM3.
Verifies that models are cached correctly, reused efficiently, and cleaned up properly.
"""

import pytest


class TestEvo2Caching:
    """Test Evo2 model instance caching."""

    def test_cache_returns_same_instance(self):
        """Verify that cache returns the same model instance for identical parameters."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        # Clear cache before test
        clear_evo2_cache()

        # Get model twice with same parameters
        model1 = get_cached_evo2_model("evo2_7b", None)
        model2 = get_cached_evo2_model("evo2_7b", None)

        # Should be the exact same object
        assert model1 is model2, "Cache should return same instance"

        # Cleanup
        clear_evo2_cache()

    def test_cache_different_models_separately(self):
        """Verify that different model configs create separate cache entries."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        clear_evo2_cache()

        # Get models with different parameters
        model1 = get_cached_evo2_model("evo2_7b", None)
        model2 = get_cached_evo2_model("evo2_7b", "/custom/path")

        # Should be different objects
        assert model1 is not model2, "Different configs should create different instances"

        clear_evo2_cache()

    def test_cached_model_not_loaded_initially(self):
        """Verify that cached models are not loaded until first call."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        clear_evo2_cache()

        model = get_cached_evo2_model("evo2_7b", None)

        # Model instance exists but not loaded
        assert model._loaded is False, "Model should not be loaded on cache retrieval"
        assert not hasattr(model, 'model') or model.model is None, "Model weights should not be loaded yet"

        clear_evo2_cache()

    @pytest.mark.uses_gpu
    def test_cached_model_lazy_loads_on_first_call(self):
        """Verify that cached model lazy loads on first inference call."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        clear_evo2_cache()

        model = get_cached_evo2_model("evo2_7b", None)
        assert model._loaded is False

        # First call triggers lazy loading
        model.sample(prompts=["ATCG"], num_tokens=10, device="cuda")

        assert model._loaded is True, "Model should be loaded after first call"
        assert model.model is not None, "Model weights should be loaded"
        assert model.device == "cuda", "Model should be on requested device"

        clear_evo2_cache()

    @pytest.mark.uses_gpu
    def test_cached_model_reuses_loaded_instance(self):
        """Verify that subsequent calls reuse the loaded model."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        clear_evo2_cache()

        # First call loads model
        model1 = get_cached_evo2_model("evo2_7b", None)
        model1.sample(prompts=["ATCG"], num_tokens=10, device="cuda")

        # Second call returns same loaded instance
        model2 = get_cached_evo2_model("evo2_7b", None)

        assert model1 is model2, "Should return same instance"
        assert model2._loaded is True, "Model should still be loaded"

        clear_evo2_cache()

    @pytest.mark.uses_gpu
    @pytest.mark.slow
    def test_device_migration_after_cpu_move(self):
        """Verify that models can be moved to CPU and back to GPU."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
        )

        clear_evo2_cache()

        model = get_cached_evo2_model("evo2_7b", None)

        # Load on GPU
        model.sample(prompts=["ATCG"], num_tokens=10, device="cuda")
        assert model.device == "cuda"

        # Move to CPU
        model.to_device("cpu")
        assert model.device == "cpu"
        assert model._loaded is True, "Model should still be loaded"

        # Move back to GPU
        model.sample(prompts=["ATCG"], num_tokens=10, device="cuda")
        assert model.device == "cuda"

        clear_evo2_cache()

    def test_clear_cache_unloads_models(self):
        """Verify that clear_cache properly unloads all models."""
        from bio_programming.tools.language_models.evo2.evo2_cache import (
            get_cached_evo2_model,
            clear_evo2_cache,
            _evo2_model_cache,
        )

        clear_evo2_cache()

        # Create cached models
        get_cached_evo2_model("evo2_7b", None)
        get_cached_evo2_model("evo2_7b", "/path1")

        assert len(_evo2_model_cache) == 2, "Should have 2 cached models"

        # Clear cache
        clear_evo2_cache()

        assert len(_evo2_model_cache) == 0, "Cache should be empty after clear"


class TestESM3Caching:
    """Test ESM3 model instance caching."""

    def test_cache_returns_same_instance(self):
        """Verify that cache returns the same ESM3 model instance."""
        from bio_programming.tools.language_models.esm3.esm3_cache import (
            get_cached_esm3_model,
            clear_esm3_cache,
        )

        clear_esm3_cache()

        model1 = get_cached_esm3_model("esm3_sm_open_v1")
        model2 = get_cached_esm3_model("esm3_sm_open_v1")

        assert model1 is model2, "Cache should return same ESM3 instance"

        clear_esm3_cache()

    def test_cached_model_not_loaded_initially(self):
        """Verify that cached ESM3 model is not loaded until first call."""
        from bio_programming.tools.language_models.esm3.esm3_cache import (
            get_cached_esm3_model,
            clear_esm3_cache,
        )

        clear_esm3_cache()

        model = get_cached_esm3_model("esm3_sm_open_v1")

        assert model._loaded is False, "ESM3 model should not be loaded on cache retrieval"
        assert not hasattr(model, 'model') or model.model is None, "ESM3 model weights should not be loaded yet"

        clear_esm3_cache()

    @pytest.mark.uses_gpu
    def test_cached_model_lazy_loads_on_first_call(self):
        """Verify that cached ESM3 model lazy loads on first inference call."""
        from bio_programming.tools.language_models.esm3.esm3_cache import (
            get_cached_esm3_model,
            clear_esm3_cache,
        )

        clear_esm3_cache()

        model = get_cached_esm3_model("esm3_sm_open_v1")
        assert model._loaded is False

        # First call triggers lazy loading
        _ = model(sequences=["MKTII"], device="cuda")

        assert model._loaded is True, "ESM3 model should be loaded after first call"
        assert model.model is not None, "ESM3 model weights should be loaded"
        assert model.device == "cuda", "ESM3 model should be on requested device"

        clear_esm3_cache()

    def test_clear_cache_unloads_models(self):
        """Verify that clear_esm3_cache properly unloads all models."""
        from bio_programming.tools.language_models.esm3.esm3_cache import (
            get_cached_esm3_model,
            clear_esm3_cache,
            _esm3_model_cache,
        )

        clear_esm3_cache()

        get_cached_esm3_model("esm3_sm_open_v1")
        assert len(_esm3_model_cache) == 1, "Should have 1 cached ESM3 model"

        clear_esm3_cache()
        assert len(_esm3_model_cache) == 0, "ESM3 cache should be empty after clear"


class TestESM2Caching:
    """Test ESM2 model instance caching."""

    def test_cache_returns_same_instance(self):
        """Verify that cache returns the same ESM2 model instance for identical parameters."""
        from bio_programming.tools.language_models.esm2.esm2_cache import (
            get_cached_esm2_model,
            clear_esm2_cache,
        )

        clear_esm2_cache()

        model1 = get_cached_esm2_model("esm2_t6_8M_UR50D")
        model2 = get_cached_esm2_model("esm2_t6_8M_UR50D")

        assert model1 is model2, "Cache should return same ESM2 instance"

        clear_esm2_cache()

    def test_cache_different_models_separately(self):
        """Verify that different ESM2 model sizes create separate cache entries."""
        from bio_programming.tools.language_models.esm2.esm2_cache import (
            get_cached_esm2_model,
            clear_esm2_cache,
        )

        clear_esm2_cache()

        model1 = get_cached_esm2_model("esm2_t6_8M_UR50D")
        model2 = get_cached_esm2_model("esm2_t12_35M_UR50D")

        assert model1 is not model2, "Different ESM2 sizes should create different instances"
        assert model1.model_checkpoint != model2.model_checkpoint

        clear_esm2_cache()

    def test_cached_model_not_loaded_initially(self):
        """Verify that cached ESM2 model is not loaded until first call."""
        from bio_programming.tools.language_models.esm2.esm2_cache import (
            get_cached_esm2_model,
            clear_esm2_cache,
        )

        clear_esm2_cache()

        model = get_cached_esm2_model("esm2_t6_8M_UR50D")

        assert model._loaded is False, "ESM2 model should not be loaded on cache retrieval"
        assert not hasattr(model, 'model') or model.model is None, "ESM2 model weights should not be loaded yet"

        clear_esm2_cache()

    @pytest.mark.uses_gpu
    def test_cached_model_lazy_loads_on_first_call(self):
        """Verify that cached ESM2 model lazy loads on first inference call."""
        from bio_programming.tools.language_models.esm2.esm2_cache import (
            get_cached_esm2_model,
            clear_esm2_cache,
        )

        clear_esm2_cache()

        model = get_cached_esm2_model("esm2_t6_8M_UR50D")
        assert model._loaded is False

        # First call triggers lazy loading
        _ = model(sequences=["MKTII"], device="cuda")

        assert model._loaded is True, "ESM2 model should be loaded after first call"
        assert model.model is not None, "ESM2 model weights should be loaded"
        assert model.device == "cuda", "ESM2 model should be on requested device"

        clear_esm2_cache()

    def test_clear_cache_unloads_models(self):
        """Verify that clear_esm2_cache properly unloads all models."""
        from bio_programming.tools.language_models.esm2.esm2_cache import (
            get_cached_esm2_model,
            clear_esm2_cache,
            _esm2_model_cache,
        )

        clear_esm2_cache()

        get_cached_esm2_model("esm2_t6_8M_UR50D")
        get_cached_esm2_model("esm2_t12_35M_UR50D")

        assert len(_esm2_model_cache) == 2, "Should have 2 cached ESM2 models"

        clear_esm2_cache()
        assert len(_esm2_model_cache) == 0, "ESM2 cache should be empty after clear"
