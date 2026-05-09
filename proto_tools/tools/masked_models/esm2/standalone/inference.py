"""Local ESM2 inference implementation."""

import json
import math
import sys
from typing import Any, Literal

import torch
from standalone_helpers import AMINO_ACIDS_LIST, get_logger, move_model_to_device, serialize_output, set_torch_seed
from tqdm import tqdm

logger = get_logger(__name__)
ESM2_MODEL_CHECKPOINTS = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


class ESM2Model:
    """ESM2 model for protein sequence embeddings and logits.

    Supports multiple model sizes from 8M to 15B parameters.
    Returns embeddings and optionally logits as tensors.
    """

    def __init__(self, model_checkpoint: ESM2_MODEL_CHECKPOINTS = "esm2_t33_650M_UR50D"):
        """Initialize ESM2 model wrapper.

        Args:
            model_checkpoint: ESM2 checkpoint (e.g., "esm2_t33_650M_UR50D")
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.tokenizer: Any = None
        self.amino_acid_token_ids: Any = None
        self.device: str | None = None
        self.model: Any = None

    def __call__(
        self,
        sequences: list[str],
        batch_size: int = 128,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
        repr_layer: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Run ESM2 inference on protein sequences.

        Args:
            sequences: Protein sequences. Each sequence must be ≤ 1022 residues
                (ESM-2's positional-encoding cap); the tool wrapper validates
                this before dispatching to the standalone.
            batch_size: Sequences per GPU forward pass. Larger batches are
                faster but use more memory.
            device: Device to run on
            verbose: Whether to print progress
            return_logits: Whether to return logits
            repr_layer: Hidden-state layer index for embeddings. ``-1`` is the
                last (top) layer.

        Returns:
            Dictionary with mean_embeddings, attention_masks, and optionally logits
        """
        # Lazy load on first call or device change
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Validate input sequences
        if not sequences:
            raise ValueError("esm2: __call__ requires at least one input sequence")
        if any(len(seq) == 0 for seq in sequences):
            raise ValueError("esm2: __call__ does not support empty sequences")

        # Get the max sequence length
        max_seq_len = max(len(seq) for seq in sequences)

        # Split the sequences into batches
        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]

        all_mean_embeddings = []
        all_logits = []
        all_attention_masks = []

        # For each batch
        for batch_sequences in tqdm(batches, desc="ESM2 inference", unit="batch", total=len(batches)):
            # Tokenize the batch
            batch_inputs = self.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            # Move the inputs to the correct device
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            # Forward pass
            with torch.inference_mode():
                batch_outputs = self.model(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    output_hidden_states=True,
                )

                # Extract embeddings from the requested hidden layer
                embeddings = batch_outputs["hidden_states"][repr_layer][:, 1:-1, :]  # Remove special tokens

                # Extract attention mask
                attention_mask = batch_inputs["attention_mask"][:, 1:-1]

                # Calculate mean embeddings
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                batch_mean_embeddings = masked_embeddings.sum(dim=1) / seq_lengths
                all_mean_embeddings.append(batch_mean_embeddings)

                # Extract logits
                logits = batch_outputs["logits"][:, 1:-1, :]  # Remove special tokens
                # Only keep the logits corresponding to the amino acids
                logits = logits[:, :, self.amino_acid_token_ids]

                # Determine padding length
                additional_padding_len = max_seq_len - embeddings.size(1)

                if additional_padding_len > 0:
                    # Pad attention_mask
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, additional_padding_len), value=0)
                    # Pad logits
                    logits = torch.nn.functional.pad(logits, (0, 0, 0, additional_padding_len), value=0.0)

                # Save attention mask and logits
                all_attention_masks.append(attention_mask)
                all_logits.append(logits)

        # Concatenate along the batch dimension
        all_mean_embeddings = torch.cat(all_mean_embeddings, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        return {
            "mean_embeddings": all_mean_embeddings,
            "logits": all_logits if return_logits else None,
            "attention_masks": all_attention_masks,
        }

    def sample(
        self,
        sequences: list[str],
        temperature: float,
        sampling_method: str = "single_pass",
        top_p: float = 1.0,
        num_steps: int = 20,
        schedule: str = "cosine",
        strategy: str = "random",
        temperature_annealing: bool = True,
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample amino acids at masked ('_') positions in protein sequences.

        Receives sequences with '_' at positions to be sampled, injects
        ``[MASK]`` tokens at those positions, runs a forward pass, and samples
        replacement amino acids from the model's predictions.
        ``iterative_refinement`` runs the MaskGIT-style loop via
        ``standalone_helpers.iterative_sampling``; ``return_logits=True`` on
        that path triggers a final forward over the completed sequences.

        Args:
            sequences (list[str]): Protein sequences with '_' at positions to sample.
            temperature (float): Sampling temperature for amino acid selection.
            sampling_method (str): "single_pass" or "iterative_refinement".
            top_p (float): Nucleus threshold (iterative only); 1.0 disables.
            num_steps (int): Refinement steps (iterative only).
            schedule (str): Unmask schedule across rounds (iterative only).
            strategy (str): Per-round commit selection (iterative only).
            temperature_annealing (bool): Anneal toward 0 across rounds (iterative only).
            batch_size (int): Sequences per GPU forward pass.
            device (str): Device to run on.
            verbose (bool): Whether to print progress.
            return_logits (bool): Whether to return per-position AA logits.
            seed (int | None): Random seed for reproducible sampling.

        Returns:
            dict[str, Any]: Dictionary with "sequences" and optionally "logits".
        """
        device_obj = torch.device(device)

        # Set torch seed for reproducibility if provided
        set_torch_seed(seed)

        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        if sampling_method == "iterative_refinement":
            return self._sample_iterative(
                sequences=sequences,
                temperature=temperature,
                top_p=top_p,
                num_steps=num_steps,
                schedule=schedule,
                strategy=strategy,
                temperature_annealing=temperature_annealing,
                batch_size=batch_size,
                device_obj=device_obj,
                verbose=verbose,
                return_logits=return_logits,
            )

        # Record mask positions, replace '_' with the model's mask token
        mask_token = self.tokenizer.mask_token
        mask_positions: list[list[int]] = []
        tokenizer_sequences: list[str] = []
        for seq in sequences:
            positions = [i for i, c in enumerate(seq) if c == "_"]
            mask_positions.append(positions)
            tokenizer_sequences.append(seq.replace("_", mask_token))

        # Batch the forward passes
        batch_ranges = [(i, min(i + batch_size, len(sequences))) for i in range(0, len(sequences), batch_size)]

        all_sampled: list[str] = []
        all_logits: list[torch.Tensor] = []

        for start, end in tqdm(
            batch_ranges,
            desc="ESM2 sampling",
            unit="batch",
            disable=not verbose,
        ):
            batch_tok_seqs = tokenizer_sequences[start:end]
            batch_masks = mask_positions[start:end]
            batch_originals = sequences[start:end]

            # Tokenize (mask tokens are handled natively by the tokenizer)
            batch_inputs = self.tokenizer(
                batch_tok_seqs,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = batch_inputs["input_ids"].to(device_obj)
            attention_mask = batch_inputs["attention_mask"].to(device_obj)

            # Forward pass
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # AA logits: remove BOS/EOS, keep only standard amino acids
                aa_logits = outputs["logits"][:, 1:-1, :][:, :, self.amino_acid_token_ids]

            # Sample at mask positions for each sequence
            for seq_idx, (orig_seq, positions) in enumerate(zip(batch_originals, batch_masks, strict=False)):
                if not positions:
                    all_sampled.append(orig_seq)
                else:
                    pos_t = torch.tensor(positions, device=device_obj)
                    pos_logits = aa_logits[seq_idx, pos_t]
                    scaled = pos_logits / max(temperature, 1e-8)
                    probs = torch.softmax(scaled, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).squeeze(-1)

                    chars = list(orig_seq)
                    for pos, aa_idx in zip(positions, sampled_idx.tolist(), strict=False):
                        chars[pos] = AMINO_ACIDS_LIST[aa_idx]
                    all_sampled.append("".join(chars))

            if return_logits:
                for seq_idx, orig_seq in enumerate(batch_originals):
                    all_logits.append(aa_logits[seq_idx, : len(orig_seq)])

        return {
            "sequences": all_sampled,
            "logits": all_logits if return_logits else None,
        }

    def _sample_iterative(
        self,
        sequences: list[str],
        temperature: float,
        top_p: float,
        num_steps: int,
        schedule: str,
        strategy: str,
        temperature_annealing: bool,
        batch_size: int,
        device_obj: torch.device,
        verbose: bool,
        return_logits: bool,
    ) -> dict[str, Any]:
        """Iterative refinement loop on top of HF ``EsmForMaskedLM``.

        Sequences are processed in chunks of ``batch_size``; each chunk runs
        the full ``num_steps`` loop as a single ``(B, L)`` tensor.
        """
        from standalone_helpers.iterative_sampling import iterative_sample

        mask_token = self.tokenizer.mask_token
        mask_token_id = int(self.tokenizer.mask_token_id)
        valid_token_ids = [int(t) for t in self.amino_acid_token_ids.tolist()]
        id_to_aa = {int(tid): aa for aa, tid in zip(AMINO_ACIDS_LIST, valid_token_ids, strict=True)}

        if verbose:
            logger.info(
                "ESM2 iterative_refinement: %d sequences x %d steps, batch_size=%d (%s, %s)",
                len(sequences),
                num_steps,
                batch_size,
                schedule,
                strategy,
            )

        sampled: list[str] = []
        all_logits: list[torch.Tensor] = []

        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            batch_inputs = self.tokenizer(
                [s.replace("_", mask_token) for s in batch_sequences],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = batch_inputs["input_ids"].to(device_obj)
            attention_mask = batch_inputs["attention_mask"].to(device_obj)

            # Iterate over the BOS/EOS-stripped interior; EOS/pad slots aren't
            # mask_token_id, so they stay inert. Allocate `full` once per chunk
            # and mutate the interior in-place each step.
            core_ids = input_ids[:, 1:-1].clone()  # (B, L-2)
            full = input_ids.clone()

            def forward_fn(
                core: torch.Tensor,
                full: torch.Tensor = full,
                attention_mask: torch.Tensor = attention_mask,
            ) -> Any:
                full[:, 1:-1] = core
                outputs = self.model(input_ids=full, attention_mask=attention_mask)
                return outputs["logits"][:, 1:-1, :]  # (B, L-2, V)

            finalized_core = iterative_sample(
                forward_fn=forward_fn,
                initial_tokens=core_ids,
                mask_token_id=mask_token_id,
                valid_token_ids=valid_token_ids,
                num_steps=num_steps,
                schedule=schedule,
                strategy=strategy,
                temperature=temperature,
                top_p=top_p,
                temperature_annealing=temperature_annealing,
            )

            # Overlay sampled AA letters at originally-`_` positions; non-AA chars (e.g. 'X') are preserved.
            for seq_idx, orig_seq in enumerate(batch_sequences):
                chars = list(orig_seq)
                for pos, c in enumerate(orig_seq):
                    if c == "_":
                        chars[pos] = id_to_aa[int(finalized_core[seq_idx, pos].item())]
                sampled.append("".join(chars))

            if return_logits:
                # Final forward on the completed chunk for per-position AA logits.
                full[:, 1:-1] = finalized_core
                with torch.inference_mode():
                    outputs = self.model(input_ids=full, attention_mask=attention_mask)
                    aa_logits = outputs["logits"][:, 1:-1, :][:, :, self.amino_acid_token_ids]  # (B, L-2, 20)
                for seq_idx, orig_seq in enumerate(batch_sequences):
                    all_logits.append(aa_logits[seq_idx, : len(orig_seq)])

        return {
            "sequences": sampled,
            "logits": all_logits if return_logits else None,
        }

    def score(
        self,
        sequences: list[str],
        batch_size: int = 32,
        device: str = "cuda",
        verbose: bool = False,
        return_logits: bool = False,
        seed: int | None = None,
    ) -> dict[str, list[Any]]:
        """Score protein sequences using ESM2 with MLM pseudo-perplexity.

        Computes pseudo-perplexity by masking each position individually and
        computing P(x_i | x_{-i}). For input of N sequences with lengths L_i,
        the total number of masked variants is sum(L_i); these are pooled
        across sequences and processed in forward-pass batches of ``batch_size``,
        for a total of ceil(sum(L_i) / batch_size) forward passes.

        Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
        calculation using the industry-standard exclusion strategy. Only positions
        with standard amino acids contribute to log-likelihood and perplexity.

        Args:
            sequences: List of protein sequences to score
            batch_size: Masked variants per forward pass, pooled across sequences.
                Larger batches are faster but use more memory.
            device: Device to run on
            verbose: Whether to print progress
            return_logits: Whether to include logits in the output
            seed: Random seed. Scoring is deterministic given the model state,
                but we still seed RNGs/cudnn flags so consecutive calls in a
                persistent worker behave identically regardless of call order.

        Returns:
            Dictionary with:
                - "logits": List of logits tensors (seq_len, vocab_size=20) if return_logits=True
                - "metrics": List of metric dicts with log_likelihood, avg_log_likelihood, perplexity
                - "vocab": List of 20 standard amino acid characters
        """
        # Lazy-load or move the model to the requested device
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        # Seed RNGs so persistent workers are call-order-independent
        set_torch_seed(seed)

        # Reject empty inputs up front
        if not sequences:
            raise ValueError("esm2: score requires at least one non-empty sequence")
        if any(len(s) == 0 for s in sequences):
            raise ValueError("esm2: score does not support empty sequences")

        # Tokenize all sequences together, padded to a common length
        encoded = self.tokenizer(sequences, add_special_tokens=True, padding=True, return_tensors="pt")
        all_input_ids = encoded["input_ids"].to(self.device)  # (N, max_len)
        all_attention_mask = encoded["attention_mask"].to(self.device)  # (N, max_len)

        # Per-sequence lengths and total variant count (one variant per residue)
        seq_lens = [len(s) for s in sequences]
        total_variants = sum(seq_lens)
        max_len = all_input_ids.shape[1]

        # Allocate the global pool: one row per (sequence, position) pair
        pooled_ids = torch.empty((total_variants, max_len), dtype=all_input_ids.dtype, device=self.device)
        pooled_attention_mask = torch.empty(
            (total_variants, max_len), dtype=all_attention_mask.dtype, device=self.device
        )
        seq_idx_per_variant = torch.empty(total_variants, dtype=torch.long, device=self.device)
        pos_idx_per_variant = torch.empty(total_variants, dtype=torch.long, device=self.device)

        # Fill the pool: copy each sequence L times, mask one position per copy
        row = 0
        for i, length in enumerate(seq_lens):
            pooled_ids[row : row + length] = all_input_ids[i].unsqueeze(0).expand(length, -1).clone()
            pooled_attention_mask[row : row + length] = all_attention_mask[i].unsqueeze(0).expand(length, -1)
            for pos in range(length):
                pooled_ids[row + pos, pos + 1] = self.tokenizer.mask_token_id  # +1 for BOS
            seq_idx_per_variant[row : row + length] = i
            pos_idx_per_variant[row : row + length] = torch.arange(length, device=self.device)
            row += length

        # Look up the true token at each masked position for log-prob scoring
        true_token_ids = all_input_ids[seq_idx_per_variant, pos_idx_per_variant + 1]

        # Per-variant validity (False at ambiguous AAs like X/B/Z — excluded from PPL)
        is_valid = torch.tensor(
            [aa in AMINO_ACIDS_LIST for seq in sequences for aa in seq],
            dtype=torch.bool,
            device=self.device,
        )

        # Output buffers indexed by global variant row
        num_aa = self.amino_acid_token_ids.shape[0]
        all_position_logits = torch.empty((total_variants, num_aa), device=self.device)
        all_true_log_probs = torch.zeros(total_variants, device=self.device)

        # Run forward passes batched across the pooled variants
        for batch_start in tqdm(
            range(0, total_variants, batch_size),
            desc="ESM2 scoring",
            unit="batch",
            disable=not verbose,
        ):
            # Slice this forward-pass batch out of the pool
            batch_end = min(batch_start + batch_size, total_variants)
            batch_ids = pooled_ids[batch_start:batch_end]
            batch_attention = pooled_attention_mask[batch_start:batch_end]

            # Forward pass over the batched masked variants
            with torch.inference_mode():
                outputs = self.model(input_ids=batch_ids, attention_mask=batch_attention)

            # Pull logits at each row's masked position only
            batch_size_actual = batch_end - batch_start
            batch_idx = torch.arange(batch_size_actual, device=self.device)
            mask_token_pos = pos_idx_per_variant[batch_start:batch_end] + 1  # +1 for BOS
            position_logits = outputs["logits"][batch_idx, mask_token_pos]  # (B, full_vocab)

            # Filter to the 20 standard AA logits and write to the global buffer
            all_position_logits[batch_start:batch_end] = position_logits[:, self.amino_acid_token_ids]

            # Record log P(true_token | context) for each row using the full-vocab softmax
            log_probs = torch.log_softmax(position_logits, dim=-1)
            batch_true = true_token_ids[batch_start:batch_end]
            all_true_log_probs[batch_start:batch_end] = log_probs[batch_idx, batch_true]

        # Aggregate the pooled outputs back into per-sequence metrics
        all_logits = []
        all_metrics = []
        cursor = 0
        for i, length in enumerate(seq_lens):
            seq_logits = all_position_logits[cursor : cursor + length]
            seq_log_probs = all_true_log_probs[cursor : cursor + length]
            seq_valid = is_valid[cursor : cursor + length]
            valid_count = int(seq_valid.sum().item())
            if valid_count == 0:
                raise ValueError(f"esm2: score sequence {i}/{len(sequences)} contains no valid amino-acid characters")
            log_prob = (seq_log_probs * seq_valid.float()).sum().item()
            avg_ll = log_prob / valid_count

            all_logits.append(seq_logits)
            all_metrics.append(
                {
                    "log_likelihood": log_prob,
                    "avg_log_likelihood": avg_ll,
                    "perplexity": math.exp(-avg_ll),
                }
            )
            cursor += length

        # Return per-sequence in input order (logits omitted unless requested)
        return {
            "logits": all_logits if return_logits else None,
            "metrics": all_metrics,
            "vocab": AMINO_ACIDS_LIST,  # Return AA-only vocab (20 tokens)
        }

    def compute_gradient(
        self,
        logits_list: list[list[float]],
        *,
        temperature: float | None = None,
        use_ste: bool = False,
        seed: int | None = None,
        backprop: bool = True,
        batch_size: int | None = None,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Chunked masked PLL gradient for relaxed ESM2 protein sequences.

        Args:
            logits_list: Input logits, shape ``(L, 20)`` in canonical amino-acid order.
            temperature: Optional softmax temperature; ``None`` uses logits as-is.
            use_ste: Use hard one-hot residues in the forward pass with soft gradients.
            seed: Random seed for deterministic worker behavior.
            backprop: If False, skip backward and return ``gradient=None``.
            batch_size: AA positions per forward pass. Defaults to 32.
            device: Execution device.
            verbose: Log progress.

        Returns:
            Dictionary with ``gradient``, mean-NLL ``loss``, metrics, and vocab.
        """
        import torch.nn.functional as F

        if not logits_list:
            raise ValueError("esm2: compute_gradient requires at least one residue")
        if any(len(row) != len(AMINO_ACIDS_LIST) for row in logits_list):
            raise ValueError(f"esm2: compute_gradient expects L x {len(AMINO_ACIDS_LIST)} logits")

        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        set_torch_seed(seed)

        if batch_size is None:
            batch_size = 32

        embed_layer = self.model.get_input_embeddings()
        cls_eos_probe = self.tokenizer("A", add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        if cls_eos_probe.numel() < 3:
            raise ValueError("esm2: tokenizer did not produce expected BOS/residue/EOS layout")
        cls_token_id = int(cls_eos_probe[0].item())
        eos_token_id = int(cls_eos_probe[-1].item())
        mask_token_id = int(self.tokenizer.mask_token_id)

        with torch.set_grad_enabled(backprop):
            seq_dist = torch.tensor(logits_list, device=self.device, dtype=torch.float32, requires_grad=backprop)
            sequence_length = int(seq_dist.shape[0])

            x = F.softmax(seq_dist / temperature, dim=-1) if temperature is not None else seq_dist
            residue_aa_idx = x.argmax(dim=-1).detach()
            residue_token_ids = self.amino_acid_token_ids[residue_aa_idx].detach()

            if use_ste:
                hard = F.one_hot(residue_aa_idx, num_classes=len(AMINO_ACIDS_LIST)).float()
                x = hard + (x - x.detach())

            residue_embeddings = x @ embed_layer.weight[self.amino_acid_token_ids]
            special_ids = torch.tensor([cls_token_id, eos_token_id], device=self.device)
            special_embeddings = embed_layer.weight[special_ids]

            input_embeddings = torch.cat(
                (special_embeddings[0:1], residue_embeddings, special_embeddings[1:2]),
                dim=0,
            ).unsqueeze(0)
            token_ids = torch.cat(
                (
                    special_ids[0:1],
                    residue_token_ids,
                    special_ids[1:2],
                ),
                dim=0,
            ).unsqueeze(0)
            attention_mask = torch.ones_like(token_ids)

            aa_positions = torch.arange(1, sequence_length + 1, device=self.device)
            seq_len = input_embeddings.shape[1]
            pos_idx = torch.arange(seq_len, device=self.device)
            mask_emb = embed_layer.weight[mask_token_id].detach()

            ie_grad = torch.zeros_like(input_embeddings) if backprop else None
            total_loss_val = 0.0
            for start in tqdm(
                range(0, sequence_length, batch_size),
                desc="ESM2 gradient",
                unit="batch",
                disable=not verbose,
            ):
                end = min(start + batch_size, sequence_length)
                chunk_aa_pos = aa_positions[start:end]
                chunk_len = end - start

                chunk_masked = pos_idx.unsqueeze(0) == chunk_aa_pos.unsqueeze(1)
                ie_chunk = input_embeddings.detach().requires_grad_(True) if backprop else input_embeddings
                chunk_input = torch.where(
                    chunk_masked.unsqueeze(-1),
                    mask_emb.view(1, 1, -1).expand(chunk_len, seq_len, -1),
                    ie_chunk.expand(chunk_len, -1, -1),
                )
                chunk_idx = torch.arange(chunk_len, device=self.device)
                chunk_token_ids = token_ids.expand(chunk_len, -1).clone()
                chunk_token_ids[chunk_idx, chunk_aa_pos] = mask_token_id

                handle = embed_layer.register_forward_hook(lambda _m, _i, _o, ci=chunk_input: ci)
                try:
                    outputs = self.model(
                        input_ids=chunk_token_ids,
                        attention_mask=attention_mask.expand(chunk_len, -1),
                    )
                finally:
                    handle.remove()
                pred = outputs["logits"][chunk_idx, chunk_aa_pos, :]
                labels = token_ids[0, chunk_aa_pos]
                loss_sum = F.cross_entropy(pred, labels, reduction="sum")
                total_loss_val += loss_sum.item()

                if backprop:
                    (loss_sum / sequence_length).backward()
                    ie_chunk_grad = ie_chunk.grad
                    if ie_chunk_grad is None:
                        raise RuntimeError("esm2: missing input-embedding gradient")
                    if ie_grad is None:
                        raise RuntimeError("esm2: missing accumulated input-embedding gradient")
                    ie_grad = ie_grad + ie_chunk_grad

        mean_nll = total_loss_val / sequence_length
        gradient_value: list[list[float]] | None = None
        if backprop:
            if ie_grad is None:
                raise RuntimeError("esm2: missing accumulated input-embedding gradient")
            (x_grad,) = torch.autograd.grad(input_embeddings, seq_dist, grad_outputs=ie_grad)
            gradient_value = x_grad.detach().cpu().tolist()

        return {
            "gradient": gradient_value,
            "loss": mean_nll,
            "metrics": {
                "log_likelihood": -total_loss_val,
                "avg_log_likelihood": -mean_nll,
                "perplexity": math.exp(mean_nll),
                "sequence_length": sequence_length,
                "model_checkpoint": self.model_checkpoint,
                "objective": "masked_pll",
            },
            "vocab": AMINO_ACIDS_LIST,
        }

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load ESM2 model and tokenizer to device."""
        if verbose:
            logger.info(f"Loading ESM2 model: {self.model_checkpoint} on {device}")

        # Load model and tokenizer
        # Uses flash attention enabled version: https://huggingface.co/fredzzp/esm2_t33_650M_UR50D
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        repo = "fredzzp/" + self.model_checkpoint
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(repo).to(device).eval()
            self.model.requires_grad_(False)
            self.tokenizer = AutoTokenizer.from_pretrained(repo)
        except OSError as e:
            raise RuntimeError(f"esm2: HF weight load from {repo!r} failed: {e}") from e

        # Create amino acid token IDs and keep them on the same device as model
        self.amino_acid_token_ids = torch.tensor(
            [self.tokenizer.get_vocab()[aa] for aa in AMINO_ACIDS_LIST],
            device=device,
        )
        self.device = device
        self._loaded = True

        if verbose:
            logger.info("ESM2 model loaded successfully")

    def to_device(self, device: str) -> None:
        """Move model to a different device."""
        if not self._loaded:
            raise ValueError("esm2: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.amino_acid_token_ids = self.amino_acid_token_ids.to(device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.amino_acid_token_ids = self.amino_acid_token_ids.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: ESM2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = ESM2Model(
            model_checkpoint=input_dict["model_checkpoint"],
        )

    operation = input_dict["operation"]
    if operation in ("embeddings", "inference"):
        return _model(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            repr_layer=input_dict["repr_layer"],
        )
    if operation == "sample":
        return _model.sample(
            sequences=input_dict["sequences"],
            temperature=input_dict["temperature"],
            sampling_method=input_dict["sampling_method"],
            top_p=input_dict["top_p"],
            num_steps=input_dict["num_steps"],
            schedule=input_dict["schedule"],
            strategy=input_dict["strategy"],
            temperature_annealing=input_dict["temperature_annealing"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            seed=input_dict["seed"],
        )
    if operation == "score":
        return _model.score(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
            return_logits=input_dict["return_logits"],
            seed=input_dict["seed"],
        )
    if operation == "compute_gradient":
        return _model.compute_gradient(
            logits_list=input_dict["logits"],
            temperature=input_dict["temperature"],
            use_ste=input_dict["use_ste"],
            seed=input_dict["seed"],
            backprop=input_dict.get("compute_gradient", True),
            batch_size=input_dict.get("batch_size"),
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )
    raise ValueError(
        f"esm2: unknown operation {operation!r}; valid: ['embeddings', 'inference', 'sample', 'score', "
        "'compute_gradient']"
    )


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esm2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
