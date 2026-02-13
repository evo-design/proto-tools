"""
Local Evo1 inference implementation.

Uses the ``evo-model`` pip package (``from evo import Evo``) and delegates
generation to ``evo.generation.generate()``.

Usage (called by EnvManager, not directly):
    python inference.py <input.json> <output.json>
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Literal, Optional

import torch

logger = logging.getLogger(__name__)

EVO1_MODEL_NAMES = Literal[
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1-8k-crispr",
    "evo-1-8k-transposon",
]


class Evo1Model:
    """
    Evo1 model implementation using the evo-model pip package.

    Example:
        >>> model = Evo1Model("evo-1-8k-base")
        >>> out = model.sample(prompts=["ATCG"], num_tokens=100)
        >>> out["sequences"]
    """

    def __init__(
        self,
        model_name: EVO1_MODEL_NAMES = "evo-1-8k-base",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def sample(
        self,
        prompts: List[str],
        num_tokens: int = 100,
        top_k: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample DNA sequences autoregressively from prompts.

        Delegates to ``evo.generation.generate()`` which supports batched
        generation with cached recurrent state.

        Args:
            prompts: DNA prompt sequences.
            num_tokens: Number of tokens to generate per prompt.
            top_k: Top-k sampling parameter.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            batch_size: Number of prompts per batch. If None, processes all at once.
            verbose: Whether to print progress.

        Returns:
            Dictionary with keys "sequences" (List[str]) and "scores" (List[float]).
        """
        if not self._loaded:
            self.load(self.device, verbose=verbose)

        if isinstance(prompts, str):
            prompts = [prompts]

        from evo.generation import generate as evo_generate

        effective_batch_size = batch_size if batch_size is not None else len(prompts)
        all_sequences: List[str] = []
        all_scores: List[float] = []

        for i in range(0, len(prompts), effective_batch_size):
            batch = prompts[i : i + effective_batch_size]
            sequences, scores = evo_generate(
                prompt_seqs=batch,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=num_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cached_generation=True,
                batched=True,
                prepend_bos=False,
                device=self.device,
                verbose=1 if verbose else 0,
            )
            all_sequences.extend(sequences)
            all_scores.extend(scores)

        assert len(all_sequences) == len(prompts)
        return {"sequences": all_sequences, "scores": all_scores}

    def load(self, device: str = "cuda", verbose: bool = False) -> None:
        """Load Evo1 model and tokenizer."""
        if verbose:
            logger.info(f"Loading Evo1: {self.model_name} on {device}")

        from evo import Evo

        evo_obj = Evo(self.model_name)
        self.model = evo_obj.model
        self.tokenizer = evo_obj.tokenizer

        self.model = self.model.to(device).eval()
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("Evo1 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model. Call load() first.")
        if self.device != device:
            self.model = self.model.to(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info("Unloading Evo1 from GPU")
            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# Entry point (called by EnvManager.call_standalone_script_in_venv)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} <input_json_path> <output_json_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    model = Evo1Model(
        model_name=input_data.get("model_name", "evo-1-8k-base"),
        device=input_data.get("device", "cuda"),
    )

    result = model.sample(
        prompts=input_data.get("prompts", []),
        num_tokens=input_data.get("num_tokens", 100),
        top_k=input_data.get("top_k", 4),
        temperature=input_data.get("temperature", 1.0),
        top_p=input_data.get("top_p", 1.0),
        batch_size=input_data.get("batch_size"),
        verbose=input_data.get("verbose", False),
    )

    with open(output_json_path, "w") as f:
        json.dump(result, f)
