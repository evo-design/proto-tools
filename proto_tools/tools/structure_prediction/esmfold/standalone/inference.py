"""ESMFold standalone inference implementation.

This script can be run independently in an isolated venv with only ESMFold
dependencies installed. It communicates via JSON files for input/output.

Usage:
    python inference.py <input_json_path> <output_json_path>
"""

import json
import logging
import sys
from contextlib import contextmanager
from typing import Any

import torch
from standalone_helpers import AMINO_ACIDS_LIST, get_logger, move_model_to_device, serialize_output

logger = get_logger(__name__)
# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ESMFold PAE head maximum: 31 bins at 1.025 A spacing, used to normalize PAE loss to [0, 1].
PAE_MAXIMUM = 31.75
OPENFOLD_AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


class ESMFoldModel:
    """ESMFold model for protein structure prediction."""

    def __init__(self) -> None:
        """Initialize ESMFold model wrapper."""
        self._loaded = False
        self.tokenizer: Any = None
        self.device: str | None = None
        self.model: Any = None

    def __call__(
        self,
        batch_data: list[dict[str, Any]],
        residue_idx_offset: int,
        chain_linker: str,
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
        num_recycles: int = 4,
    ) -> list[dict[str, Any]]:
        """Run ESMFold structure prediction on protein sequences.

        Args:
            batch_data: List of dicts with keys: linked_seq, chains, seq_lengths
            residue_idx_offset: Offset between chains in residue numbering
            chain_linker: Sequence used to link chains
            device: Device to run on
            verbose: Whether to print status messages
            include_pae_matrix: Attach the full per-residue PAE matrix.
            num_recycles: Iterative refinement passes through ESMFold (training default 4).

        Returns:
            List of dicts with keys: pdb, avg_plddt, ptm
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Extract sequences
        linked_sequences = [item["linked_seq"] for item in batch_data]

        if verbose:
            logger.info(f"Starting ESMFold inference on {len(batch_data)} structure(s)...")

        # Enable trunk chunking for long sequences
        max_seq_len = max(len(seq) for seq in linked_sequences)
        if max_seq_len > 1200:
            if verbose:
                logger.info(f"Long sequence detected ({max_seq_len} residues), enabling trunk chunking (chunk_size=64)")
            self.model.trunk.set_chunk_size(64)

        # Forward-only path; avoid inference tensors.
        with torch.no_grad(), _allow_tf32():
            # Tokenize all sequences
            tokenized_inputs = self.tokenizer(
                linked_sequences, return_tensors="pt", padding=True, add_special_tokens=False
            )
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

            # Build position_ids and linker_masks
            position_ids, linker_masks = self._build_batch_tensors(batch_data, residue_idx_offset, chain_linker)
            tokenized_inputs["position_ids"] = position_ids

            # Forward pass
            logger.update_status(f"Folding {len(batch_data)} complex(es), num_recycles={num_recycles}")
            outputs = self.model(**tokenized_inputs, num_recycles=num_recycles)

            # Apply linker masking
            outputs["atom37_atom_exists"] = outputs["atom37_atom_exists"] * linker_masks[:, :, None]

        # Extract per-complex results
        return [self._extract_result(outputs, idx, include_pae_matrix) for idx in range(len(batch_data))]

    def compute_gradient(
        self,
        *,
        complex_data: dict[str, Any],
        logits_list: list[list[float]],
        target_chain_indices: list[int],
        residue_idx_offset: int,
        chain_linker: str,
        temperature: float = 1.0,
        soft: float = 1.0,
        hard: float = 0.0,
        loss_weights: dict[str, float] | None = None,
        compute_gradient: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        include_pae_matrix: bool = False,
        num_recycles: int = 4,
    ) -> dict[str, Any]:
        """Run one differentiable ESMFold confidence pass for a target chain.

        The discrete ESMFold ``aatype`` path still receives the hard argmax
        sequence required for atom masks and PDB export. The ESM language-model
        embeddings and ESMFold amino-acid embedding receive a relaxed
        distribution, so confidence losses can differentiate back to the input
        logits.
        """
        if loss_weights is None:
            loss_weights = {"plddt": 1.0}
        unknown = set(loss_weights) - {"plddt", "ptm", "pae"}
        if unknown:
            raise ValueError(f"esmfold: unknown loss_weights keys {sorted(unknown)}")

        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)
        active_device = self.device
        if active_device is None:
            raise RuntimeError("ESMFold model device is not initialized.")

        chains = list(complex_data["chains"])
        logits = torch.tensor(logits_list, device=active_device, dtype=torch.float32, requires_grad=compute_gradient)
        decoded = _decode_logits(logits)
        for chain_idx in target_chain_indices:
            chains[chain_idx] = decoded

        complex_data = {
            **complex_data,
            "chains": chains,
            "linked_seq": chain_linker.join(chains),
            "seq_lengths": [len(chain) for chain in chains],
        }
        max_seq_len = len(complex_data["linked_seq"])
        if max_seq_len > 1200:
            if verbose:
                logger.info(f"Long sequence detected ({max_seq_len} residues), enabling trunk chunking (chunk_size=64)")
            self.model.trunk.set_chunk_size(64)

        if all(weight == 0.0 for weight in loss_weights.values()):
            # Forward-only path; avoid inference tensors.
            with torch.no_grad(), _allow_tf32():
                outputs = self._run_discrete_forward(
                    complex_data,
                    residue_idx_offset=residue_idx_offset,
                    chain_linker=chain_linker,
                    num_recycles=num_recycles,
                )
            result = self._extract_result(outputs, 0, include_pae_matrix)
            metrics = _metrics_with_losses(result, {})
            return {
                "gradient": serialize_output(torch.zeros_like(logits)) if compute_gradient else None,
                "loss": 0.0,
                "metrics": metrics,
                "vocab": AMINO_ACIDS_LIST,
                "pdb": result["pdb"],
            }

        # Differentiable path; tensors must be saved for backward.
        with torch.inference_mode(False), torch.set_grad_enabled(compute_gradient), _allow_tf32():
            relaxed_probs = _relaxed_amino_acid_probs(logits, temperature=temperature, soft=soft, hard=hard)
            outputs = self._run_relaxed_forward(
                complex_data,
                relaxed_probs=relaxed_probs,
                target_chain_indices=target_chain_indices,
                residue_idx_offset=residue_idx_offset,
                chain_linker=chain_linker,
                num_recycles=num_recycles,
            )
            loss_terms = self._confidence_loss_terms(outputs)
            missing = [key for key, weight in loss_weights.items() if weight != 0.0 and key not in loss_terms]
            if missing:
                raise RuntimeError(
                    f"esmfold: model output did not include required confidence terms {missing}; "
                    f"available terms: {sorted(loss_terms)}"
                )
            weighted_terms = [float(weight) * loss_terms[key] for key, weight in loss_weights.items() if weight != 0.0]
            weighted_loss = torch.stack(weighted_terms).sum()

            gradient = None
            if compute_gradient:
                gradient = torch.autograd.grad(weighted_loss, logits, allow_unused=False)[0]

        result = self._extract_result(outputs, 0, include_pae_matrix)
        metrics = _metrics_with_losses(result, {key: float(loss_terms[key].detach()) for key in loss_terms})
        return {
            "gradient": serialize_output(gradient) if gradient is not None else None,
            "loss": float(weighted_loss.detach()),
            "metrics": metrics,
            "vocab": AMINO_ACIDS_LIST,
            "pdb": result["pdb"],
        }

    def _run_discrete_forward(
        self,
        complex_data: dict[str, Any],
        *,
        residue_idx_offset: int,
        chain_linker: str,
        num_recycles: int,
    ) -> dict[str, torch.Tensor | None]:
        """Run the normal hard-token model path for one prepared complex."""
        tokenized_inputs = self.tokenizer(
            [complex_data["linked_seq"]], return_tensors="pt", padding=True, add_special_tokens=False
        )
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        position_ids, linker_masks = self._build_batch_tensors([complex_data], residue_idx_offset, chain_linker)
        tokenized_inputs["position_ids"] = position_ids
        outputs = self.model(**tokenized_inputs, num_recycles=num_recycles)
        outputs["atom37_atom_exists"] = outputs["atom37_atom_exists"] * linker_masks[:, :, None]
        return dict(outputs.items())

    def _run_relaxed_forward(
        self,
        complex_data: dict[str, Any],
        *,
        relaxed_probs: torch.Tensor,
        target_chain_indices: list[int],
        residue_idx_offset: int,
        chain_linker: str,
        num_recycles: int,
    ) -> dict[str, torch.Tensor | None]:
        """Run ESMFold with relaxed target-chain embeddings."""
        chains = complex_data["chains"]
        linked_seq = complex_data["linked_seq"]
        device = self.device
        if device is None:
            raise RuntimeError("ESMFold model device is not initialized.")
        aa = _sequence_to_aatype(linked_seq, device).unsqueeze(0)
        attention_mask = torch.ones_like(aa, device=device)
        position_ids, linker_masks = self._build_batch_tensors([complex_data], residue_idx_offset, chain_linker)
        target_spans = _chain_spans(chains, chain_linker, target_chain_indices)

        outputs = self._forward_with_relaxed_embeddings(
            aa=aa,
            attention_mask=attention_mask,
            position_ids=position_ids,
            relaxed_probs=relaxed_probs,
            target_spans=target_spans,
            num_recycles=num_recycles,
        )
        atom37_atom_exists = outputs["atom37_atom_exists"]
        assert atom37_atom_exists is not None
        outputs["atom37_atom_exists"] = atom37_atom_exists * linker_masks[:, :, None]
        return outputs

    def _forward_with_relaxed_embeddings(
        self,
        *,
        aa: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        relaxed_probs: torch.Tensor,
        target_spans: list[tuple[int, int]],
        num_recycles: int,
    ) -> dict[str, torch.Tensor | None]:
        """ESMFold forward pass with differentiable target-chain embeddings."""
        from transformers.models.esm.modeling_esmfold import categorical_lddt
        from transformers.models.esm.openfold_utils import (
            compute_predicted_aligned_error,
            compute_tm,
            make_atom14_masks,
        )

        cfg = self.model.config.esmfold_config
        batch_size, seq_len = aa.shape

        esmaa = self.model.af2_idx_to_esm_idx(aa, attention_mask)
        bos = esmaa.new_full((batch_size, 1), self.model.esm_dict_cls_idx)
        eos = esmaa.new_full((batch_size, 1), self.model.esm_dict_padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        esmaa[range(batch_size), (esmaa != self.model.esm_dict_padding_idx).sum(1)] = self.model.esm_dict_eos_idx

        esm_attention_mask = esmaa != self.model.esm_dict_padding_idx
        token_embeddings = self.model.esm.embeddings.word_embeddings(esmaa)
        esm_ids = torch.tensor(
            [self.model.config.vocab_list.index(aa_symbol) for aa_symbol in AMINO_ACIDS_LIST],
            device=self.device,
        )
        relaxed_esm_embeddings = relaxed_probs.to(token_embeddings.dtype) @ self.model.esm.embeddings.word_embeddings(
            esm_ids
        )
        for start, end in target_spans:
            token_embeddings[0, start + 1 : end + 1] = relaxed_esm_embeddings

        embedding_output = self.model.esm.embeddings(
            input_ids=esmaa,
            attention_mask=esm_attention_mask,
            inputs_embeds=token_embeddings,
        )
        esm_outputs = self.model.esm(
            inputs_embeds=embedding_output,
            attention_mask=esm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        esm_s = torch.stack(esm_outputs["hidden_states"], dim=2)[:, 1:-1]
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)

        s_z_0 = s_s_0.new_zeros(batch_size, seq_len, seq_len, cfg.trunk.pairwise_state_dim)
        if self.model.config.esmfold_config.embed_aa:
            aa_embeddings = self.model.embedding(aa)
            af2_ids = torch.tensor(
                [OPENFOLD_AMINO_ACIDS.index(aa_symbol) + 1 for aa_symbol in AMINO_ACIDS_LIST],
                device=self.device,
            )
            relaxed_aa_embeddings = relaxed_probs.to(aa_embeddings.dtype) @ self.model.embedding(af2_ids)
            for start, end in target_spans:
                aa_embeddings[0, start:end] = relaxed_aa_embeddings
            s_s_0 = s_s_0 + aa_embeddings

        structure = self.model.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
        structure = {
            key: value
            for key, value in structure.items()
            if key
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }
        disto_logits = self.model.distogram_head(structure["s_z"])
        structure["distogram_logits"] = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["lm_logits"] = self.model.lm_head(structure["s_s"])

        structure["aatype"] = aa
        make_atom14_masks(structure)
        for key in ["atom14_atom_exists", "atom37_atom_exists"]:
            structure[key] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = position_ids

        lddt_head = self.model.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0],
            batch_size,
            seq_len,
            -1,
            self.model.lddt_bins,
        )
        structure["lddt_head"] = lddt_head
        structure["plddt"] = categorical_lddt(lddt_head[-1], bins=self.model.lddt_bins)

        ptm_logits = self.model.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.model.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.model.distogram_bins))
        return structure

    def _confidence_loss_terms(self, outputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
        """Return differentiable ESMFold confidence losses on the masked complex."""
        atom_exists = outputs["atom37_atom_exists"]
        plddt = outputs["plddt"]
        assert atom_exists is not None and plddt is not None
        avg_plddt = (plddt * atom_exists).sum() / atom_exists.sum().clamp(min=1)

        loss_terms: dict[str, torch.Tensor] = {"plddt": 1.0 - avg_plddt}
        ptm = outputs.get("ptm")
        if ptm is not None:
            loss_terms["ptm"] = 1.0 - ptm.reshape(-1)[0]
        pae = outputs.get("predicted_aligned_error")
        if pae is not None:
            residue_mask = atom_exists.any(dim=-1)
            pae_mask = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)
            avg_pae = (pae * pae_mask).sum() / pae_mask.sum().clamp(min=1)
            loss_terms["pae"] = torch.clamp(avg_pae / PAE_MAXIMUM, max=1.0)
        return loss_terms

    def _build_batch_tensors(
        self, batch_data: list[dict[str, Any]], residue_idx_offset: int, chain_linker: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build position_ids and linker_masks for entire batch.

        Returns:
            position_ids: (batch_size, max_length) - residue numbering with offsets
            linker_masks: (batch_size, max_length) - 0 for linkers, 1 for residues
        """
        batch_size = len(batch_data)
        max_length = max(len(item["linked_seq"]) for item in batch_data)

        position_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        linker_masks = torch.zeros(batch_size, max_length, dtype=torch.float32, device=self.device)

        for batch_idx, item in enumerate(batch_data):
            seq_len = len(item["linked_seq"])

            # Build tensors for this complex
            pos_ids, mask = self._build_single_tensors(item["chains"], residue_idx_offset, chain_linker)

            # Fill batch tensors (rest stays 0/1 for padding)
            position_ids[batch_idx, :seq_len] = pos_ids
            linker_masks[batch_idx, :seq_len] = mask

        return position_ids, linker_masks

    def _build_single_tensors(
        self, chains: list[str], residue_idx_offset: int, chain_linker: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build position_ids and linker_mask for a single complex.

        Args:
            chains: List of chain sequences
            residue_idx_offset: Offset between chains in residue numbering
            chain_linker: Sequence used to link chains

        Returns:
            Tuple of (position_ids, linker_mask) tensors
        """
        # Calculate total length
        seq_length = sum(len(chain) for chain in chains)
        if len(chains) > 1:
            seq_length += len(chain_linker) * (len(chains) - 1)

        # Initialize
        position_ids = torch.arange(seq_length, device=self.device)
        linker_mask = torch.ones(seq_length, dtype=torch.float32, device=self.device)

        # Apply residue offsets and mask linkers
        pos = 0
        for chain_idx, chain in enumerate(chains):
            chain_len = len(chain)

            # Apply offset to this chain (only if offset > 0)
            if residue_idx_offset > 0:
                position_ids[pos : pos + chain_len] += chain_idx * residue_idx_offset
            pos += chain_len

            # Mask linker (if not last chain) - ALWAYS do this regardless of offset
            if chain_idx < len(chains) - 1:
                linker_len = len(chain_linker)
                linker_mask[pos : pos + linker_len] = 0
                pos += linker_len

        return position_ids, linker_mask

    def _extract_result(
        self, outputs: dict[str, torch.Tensor | None], batch_idx: int, include_pae_matrix: bool = False
    ) -> dict[str, Any]:
        """Extract results for a single complex from batched outputs.

        Returns:
            Dict with keys: pdb, avg_plddt, ptm, avg_pae
        """
        # Structure module tensors have shape (num_blocks, batch, ...) - batch is dim 1.
        # All other tensors have shape (batch, ...) - batch is dim 0.
        structure_module_tensors = {
            "positions",
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "states",
            "lddt_head",
        }
        complex_output: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if value is None:
                continue
            if value.ndim == 0:  # Scalar
                complex_output[key] = value.detach()
            elif key in structure_module_tensors:
                # structure module tensors have batch as dim 1
                complex_output[key] = value[:, batch_idx : batch_idx + 1, ...].detach()
            else:
                # all other tensors have batch as dim 0
                complex_output[key] = value[batch_idx : batch_idx + 1].detach()

        # Convert to PDB
        pdb_output = self.model.output_to_pdb(complex_output)[0]

        # Calculate average pLDDT
        atom_exists = complex_output["atom37_atom_exists"]
        plddt = complex_output["plddt"]
        avg_plddt = ((plddt * atom_exists).sum() / atom_exists.sum().clamp(min=1)).item()

        # Extract PTM score
        ptm_tensor = complex_output.get("ptm")
        ptm = ptm_tensor.item() if ptm_tensor is not None else None

        # Calculate average pAE (masked to exclude padding)
        pae = complex_output.get("predicted_aligned_error")
        if pae is not None:
            # Create 1D mask from atom_exists (any atom existing means residue is valid)
            residue_mask = atom_exists.any(dim=-1)  # (1, seq_len)
            # Create 2D mask for valid (i, j) residue pairs
            pae_mask = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)  # (1, seq_len, seq_len)
            avg_pae = float((pae * pae_mask).sum() / pae_mask.sum().clamp(min=1))
            pae = pae[0].tolist() if include_pae_matrix else None
        else:
            avg_pae = None
            pae = None

        return {
            "pdb": pdb_output,
            "avg_plddt": float(avg_plddt),
            "ptm": float(ptm) if ptm is not None else None,
            "avg_pae": avg_pae,
            "pae": pae,
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load ESMFold model and tokenizer to device."""
        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError:
            raise ImportError(
                "esmfold: transformers not importable; ensure ESMFold dependencies are installed"
            ) from None

        logger.update_status(f"Loading ESMFold model: facebook/esmfold_v1 on {device}")

        repo = "facebook/esmfold_v1"
        try:
            self.model = EsmForProteinFolding.from_pretrained(repo, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(repo)
        except OSError as e:
            raise RuntimeError(f"esmfold: HF weight load from {repo!r} failed: {e}") from e

        logger.update_status(f"Moving ESMFold to {device} (fp16)")
        self.model = self.model.to(device)
        self.model.esm = self.model.esm.half()
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESMFold model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("esmfold: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@contextmanager  # type: ignore[arg-type]
def _allow_tf32() -> None:  # type: ignore[misc]
    """Temporarily enable TF32 for matmul operations."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _decode_logits(logits: torch.Tensor) -> str:
    """Decode proto-order amino-acid logits to a hard sequence."""
    indices = logits.detach().argmax(dim=-1).tolist()
    return "".join(AMINO_ACIDS_LIST[int(index)] for index in indices)


def _relaxed_amino_acid_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    soft: float,
    hard: float,
) -> torch.Tensor:
    """Convert logits to relaxed amino-acid probabilities with optional STE."""
    import torch.nn.functional as F

    probs = F.softmax(logits / temperature, dim=-1)
    hard_probs = F.one_hot(probs.argmax(dim=-1), num_classes=len(AMINO_ACIDS_LIST)).to(probs.dtype)
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    if hard > 0.0:
        straight_through = hard_probs + probs - probs.detach()
        relaxed = hard * straight_through + (1.0 - hard) * relaxed
    return relaxed


def _sequence_to_aatype(sequence: str, device: str | torch.device) -> torch.Tensor:
    """Convert a protein sequence to OpenFold/ESMFold residue indices."""
    mapping = {aa: idx for idx, aa in enumerate(OPENFOLD_AMINO_ACIDS)}
    x_index = len(OPENFOLD_AMINO_ACIDS)
    return torch.tensor([mapping.get(aa, x_index) for aa in sequence], device=device, dtype=torch.long)


def _chain_spans(chains: list[str], chain_linker: str, selected_indices: list[int]) -> list[tuple[int, int]]:
    """Return linked-sequence half-open spans for selected chain indices."""
    selected = set(selected_indices)
    spans: list[tuple[int, int]] = []
    pos = 0
    for chain_idx, chain in enumerate(chains):
        start = pos
        end = start + len(chain)
        if chain_idx in selected:
            spans.append((start, end))
        pos = end
        if chain_idx < len(chains) - 1:
            pos += len(chain_linker)
    return spans


def _metrics_with_losses(result: dict[str, Any], loss_terms: dict[str, float]) -> dict[str, Any]:
    """Build JSON-safe ESMFold metrics with unweighted per-term losses."""
    metrics = {
        "avg_plddt": result["avg_plddt"],
        "ptm": result["ptm"],
        "avg_pae": result["avg_pae"],
        "pae": result.get("pae"),
    }
    metrics.update({f"loss_{key}": value for key, value in loss_terms.items()})
    return metrics


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMFoldModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESMFoldModel()

    operation = input_dict["operation"]
    if operation == "predict":
        results = _model(
            batch_data=input_dict["batch_data"],
            residue_idx_offset=input_dict["residue_idx_offset"],
            chain_linker=input_dict["chain_linker"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
            num_recycles=input_dict["num_recycles"],
        )
        return {"results": results}
    if operation == "compute_gradient":
        return _model.compute_gradient(
            complex_data=input_dict["complex_data"],
            logits_list=input_dict["logits"],
            target_chain_indices=input_dict["target_chain_indices"],
            residue_idx_offset=input_dict["residue_idx_offset"],
            chain_linker=input_dict["chain_linker"],
            temperature=input_dict["temperature"],
            soft=input_dict["soft"],
            hard=input_dict["hard"],
            loss_weights=input_dict["loss_weights"],
            compute_gradient=input_dict["compute_gradient"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            include_pae_matrix=input_dict["include_pae_matrix"],
            num_recycles=input_dict["num_recycles"],
        )
    raise ValueError(f"esmfold: unknown operation {operation!r}; valid: ['predict', 'compute_gradient']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    # Model not loaded yet - will use device on next call
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    from standalone_helpers import get_pytorch_memory_stats

    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


# ============================================================================
# Standalone Script Entry Point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmfold: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
