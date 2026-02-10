from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpliceTransformerModel:
    def __init__(self, context_length: int = 4000):
        self.context_length = context_length
        self.device = None
        self.model = None
        self._loaded = False

    def __call__(
        self,
        target_seqs: List[str],
        left_contexts: List[str],
        right_contexts: List[str],
        device: str = "cuda",
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run SpliceTransformer inference on sequences with contexts.

        Args:
            target_seqs: Target sequences to make predictions on
            left_contexts: Left context sequences (must be context_length long)
            right_contexts: Right context sequences (must be context_length long)
            device: Device to run inference on
            verbose: Whether to print status messages

        Returns:
            Predictions of shape (batch, target_length, 18)
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        assert len(target_seqs) == len(left_contexts) == len(right_contexts), \
            "Number of targets must be the same as the number of left and right contexts"

        seqs_tokenized = []
        for target, left, right in zip(target_seqs, left_contexts, right_contexts):
            assert len(left) == len(right) == self.context_length, \
                f"Length of left and right contexts must be {self.context_length}, got {len(left)} and {len(right)}"
            seq = left + target + right
            seqs_tokenized.append(self._one_hot_encode(seq))
        seqs_tokenized = np.stack(seqs_tokenized)

        prediction = self._calc_batched_sequence(seqs_tokenized)  # (batch, target_length, 18)
        return prediction


    def _one_hot_encode(self, seq: str):
        """
        Parse input RNA sequence into one-hot-encoding format
        """
        IN_MAP = np.asarray(
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        seq = seq.upper().replace('A', '1').replace('C', '2')
        seq = seq.replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        seq = np.asarray(list(map(int, list(seq))))
        seq = IN_MAP[seq.astype('int8')]
        return seq


    def _post_decorate(self, outputs: torch.Tensor):
        outputs[:, :3, :] = torch.nn.functional.softmax(outputs[:, :3, :], dim=1)
        outputs[:, 3:, :] = torch.sigmoid(outputs[:, 3:, :])
        return outputs


    def _step(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass.

        Args:
            inputs: Encoded sequence tensor of shape (batch, 4, length)

        Returns:
            Model output tensor
        """
        assert len(inputs.size()) == 3
        with torch.no_grad():
            out = self.model(inputs).cpu().detach()
            out = self._post_decorate(out)
            return out


    def _calc_single_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        Calculate model output for a single sequence.

        Args:
            seq: One-hot encoded sequence array of shape (length, 4)

        Returns:
            Model output array
        """
        seq = torch.tensor(seq).to(self.device)
        seq = seq.unsqueeze(0).transpose(1, 2)
        res = self._step(seq.float())
        res = res[0].transpose(0, 1).numpy()
        return res


    def _calc_batched_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        Calculate model output for multiple sequences.

        Args:
            seq: One-hot encoded sequence array of shape (batch, length, 4)

        Returns:
            Model output array
        """
        seq = torch.tensor(seq).to(self.device)
        seq = seq.transpose(1, 2)  # 　(batch, length, 4) -> (batch, 4, length)
        res = self._step(seq.float())
        res = res.transpose(1, 2).numpy()  # 　(batch, 4, length) -> (batch, length, 4)
        return res

    # ============================================================================
    # Model Loading & Device Management
    # ============================================================================
    def _fix_state_dict_keys(self, state_dict: dict) -> dict:
        """Fix mismatched weight keys in checkpoint."""
        return {
            key.replace("attn.pos_emb.weights_", "attn.pos_emb.weights."): value
            for key, value in state_dict.items()
        }

    def load(self, device: str = "cuda", verbose: bool = False) -> None:
        """Load SpliceTransformer model to device."""
        logger.debug(f"Loading SpliceTransformer (context_length={self.context_length}) on {device}")

        from model import SpTransformer

        self.model = SpTransformer(
            128,
            context_len=self.context_length,
            tissue_num=15,
            max_seq_len=8192,
            attn_depth=8,
            training=False,
        ).to(device).eval()

        # Download checkpoint using HuggingFace Hub (respects HF_HOME env var)
        logger.debug("Downloading SpliceTransformer checkpoint from HuggingFace Hub...")

        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="brianhie/SpTransformer",
            filename="SpTransformer_pytorch.ckpt",
        )

        # Load and fix state dict
        save_dict = torch.load(model_path, map_location=device)
        state_dict = self._fix_state_dict_keys(save_dict["state_dict"])
        self.model.load_state_dict(state_dict)

        self.device = device
        self._loaded = True

        logger.debug("SpliceTransformer model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise RuntimeError("Cannot move unloaded model to device. Call load() first.")

        if self.device != device:
            self.model = self.model.to(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            logger.debug("Unloading SpliceTransformer from GPU")

            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
