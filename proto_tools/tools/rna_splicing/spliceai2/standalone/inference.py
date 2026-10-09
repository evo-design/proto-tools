"""Standalone inference entry point for SpliceAI2 splice and variant-effect prediction."""

import json
import os
import sys
from typing import Any

from standalone_helpers import get_logger, move_model_to_device, resolve_weights_dir, serialize_output

logger = get_logger(__name__)

# The two ensemble checkpoints, downloaded by setup.sh into the managed weights directory.
CHECKPOINTS = ("spliceai2_multispecies_13M_01", "spliceai2_multispecies_13M_02")

# Context cropped from each side of the input; in sync with shared_data_models.SPLICEAI2_FLANK.
FLANK = 65_536
# Variant-scoring window (196,608 bp, reporting the central 65,536) and shared candidate sites.
VARIANT_INPUT_LENGTH = 196_608
VARIANT_CANDIDATE_SITES = 1024
# Predicted output length must be a multiple of the deepest pooling factor.
LENGTH_MULTIPLE = 16
# Sites above this usage become candidates for junction and transcript prediction.
CANDIDATE_SITE_THRESHOLD = 0.01

# Marks padding, which encodes as all-zero nucleotide channels (upstream's out-of-bounds input).
PAD = "."

SITE_EFFECTS = ("donor_gain", "donor_loss", "acceptor_gain", "acceptor_loss")


def _encode(sequence: str, assembly: str, reverse_complement: bool = False) -> Any:
    """One-hot encode a sequence with its species channel; padding stays all-zero."""
    import numpy as np
    from spliceai2.dataset import _one_hot_encode

    x = _one_hot_encode(sequence, assembly)
    x[:4, np.frombuffer(sequence.encode(), dtype=np.uint8) == ord(PAD)] = 0
    if reverse_complement:
        x[:4] = x[:4][::-1, ::-1]
    return x.astype("float32")


def _fasta_chrom(fasta: Any, chrom: str) -> str:
    """Name ``chrom`` as the FASTA does, adding or dropping a ``chr`` prefix (``chr1`` and ``1`` both match)."""
    if chrom in fasta:
        return chrom
    alternate = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alternate in fasta:
        return alternate
    raise ValueError(f"spliceai2: chromosome {chrom!r} is not in the reference FASTA")


class SpliceAI2Model:
    """Manages the SpliceAI2 two-model ensemble and its inference."""

    def __init__(self) -> None:
        """Initialize SpliceAI2Model."""
        self.device: str | None = None
        self.models: list[Any] = []
        self._fastas: dict[str, Any] = {}
        self._loaded = False

    # ============================================================================
    # Model Loading & Device Management
    # ============================================================================
    def load(self, device: str) -> None:
        """Load both ensemble checkpoints onto device."""
        from spliceai2.model import SpliceAI2

        weights_dir = resolve_weights_dir("spliceai2")
        logger.update_status("Loading SpliceAI2 checkpoints")
        self.models = [
            SpliceAI2.load_from_checkpoint(os.path.join(weights_dir, name, "model.ckpt"), map_location=device).eval()
            for name in CHECKPOINTS
        ]
        self.device = device
        self._loaded = True
        logger.debug("SpliceAI2 models loaded on %s", device)

    def _ensure_loaded(self, device: str) -> None:
        """Load the ensemble, or move it when the requested device changed."""
        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move both loaded models to a different device."""
        if not self._loaded:
            raise ValueError("spliceai2: cannot move unloaded model to device; call load() first")
        if self.device == device:
            return
        self.models = [move_model_to_device(model, self.device, device) for model in self.models]
        self.device = device

    def _fasta(self, path: str) -> Any:
        """Open (and cache) a reference genome FASTA."""
        import pyfaidx

        if path not in self._fastas:
            self._fastas[path] = pyfaidx.Fasta(path)
        return self._fastas[path]

    # ============================================================================
    # Variant Scoring
    # ============================================================================
    def _variant_window(self, fasta: Any, variant: dict[str, Any]) -> tuple[str, str]:
        """Reference and alternate input windows centered on the variant, padded past chromosome ends."""
        pos, ref, alt = variant["position"], variant["ref"], variant["alt"]
        chrom = _fasta_chrom(fasta, variant["chromosome"])
        chrom_len = len(fasta[chrom])
        if pos + len(ref) - 1 > chrom_len:
            raise ValueError(f"spliceai2: {chrom}:{pos} lies beyond the end of chromosome {chrom} ({chrom_len} bp)")

        # Upstream's window: half the input on each side, extended by any deleted bases.
        half = VARIANT_INPUT_LENGTH // 2
        start = pos - 1 - half
        end = pos - 1 + half + max(0, len(ref) - len(alt))
        sequence = fasta[chrom][max(0, start) : min(end, chrom_len)].seq.upper()
        window = PAD * max(0, -start) + sequence + PAD * max(0, end - chrom_len)

        observed = window[half : half + len(ref)]
        if observed != ref:
            raise ValueError(
                f"spliceai2: ref {ref!r} does not match the reference genome at {chrom}:{pos} (found {observed!r})"
            )
        ref_window = window[:VARIANT_INPUT_LENGTH]
        alt_window = (window[:half] + alt + window[half + len(ref) :])[:VARIANT_INPUT_LENGTH]
        return ref_window, alt_window

    def score(
        self,
        variants: list[dict[str, Any]],
        reference_fasta: str,
        assembly: str,
        device: str,
    ) -> list[dict[str, Any]]:
        """Score variants, returning one row keyed by upstream column names per variant."""
        import torch
        from spliceai2.model import VariantAnnotator

        self._ensure_loaded(device)
        fasta = self._fasta(reference_fasta)
        annotator = VariantAnnotator(self.models, VARIANT_CANDIDATE_SITES)

        rows: list[dict[str, Any]] = []
        for i, variant in enumerate(variants):
            logger.update_status(f"Scoring variant {i + 1}/{len(variants)}")
            ref_window, alt_window = self._variant_window(fasta, variant)
            minus = variant["strand"] == "-"
            batch = (
                torch.from_numpy(_encode(ref_window, assembly, minus))[None].to(device),
                torch.from_numpy(_encode(alt_window, assembly, minus))[None].to(device),
                torch.tensor([-1 if minus else 1], device=device),
                torch.tensor([len(variant["ref"])], device=device),
                torch.tensor([len(variant["alt"])], device=device),
            )
            # Upstream scores under 16-bit mixed precision.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                result, columns = annotator.predict_step(batch)
            values = result[0].float().cpu().tolist()
            row = {col: (int(v) if "dist" in col else v) for col, v in zip(columns, values, strict=True)}
            row["spliceai2_summary_score"] = max(row[f"{effect}_delta_score_0"] for effect in SITE_EFFECTS)
            rows.append(row)
        return rows

    # ============================================================================
    # Splice and Transcript Prediction
    # ============================================================================
    def _predict_one(self, sequence: str, assembly: str, num_transcripts: int) -> dict[str, Any]:
        """Predict splice sites, junctions, and transcripts for one sense-strand sequence."""
        import torch
        from spliceai2.utils import decode_topk_tx

        n = len(sequence)
        tail = -n % LENGTH_MULTIPLE
        padded = PAD * FLANK + sequence + PAD * (FLANK + tail)
        x = torch.from_numpy(_encode(padded, assembly))[None].to(self.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ftrs_list = [model.forward(x) for model in self.models]
            out_ss = torch.stack([model.forward_1d(ftrs) for model, ftrs in zip(self.models, ftrs_list, strict=True)])
            out_ss = out_ss.mean(dim=0).float()[:, :, :n]

            # Candidate sites: every position whose donor or acceptor usage clears the threshold.
            site_logits = out_ss.amax(dim=1)
            num_candidates = int((site_logits.sigmoid() > CANDIDATE_SITE_THRESHOLD).sum())
            probabilities = out_ss[0].sigmoid().cpu().numpy()
            prediction: dict[str, Any] = {
                "donor_probabilities": probabilities[0],
                "acceptor_probabilities": probabilities[1],
                "junctions": [],
                "transcripts": [],
            }
            if num_candidates < 2:
                return prediction

            idxs = site_logits.topk(num_candidates, dim=1)[1].sort(dim=1).values
            out_jxn = torch.stack(
                [model.forward_2d(ftrs, idxs) for model, ftrs in zip(self.models, ftrs_list, strict=True)]
            )
            out_jxn = out_jxn.mean(dim=0).float()
            out_tx = out_ss[:, 0].gather(1, idxs).unsqueeze(2) + out_jxn + out_ss[:, 1].gather(1, idxs).unsqueeze(1)

        sites = idxs[0].cpu().numpy()
        # The lower triangle is masked to a large negative logit, so only donor < acceptor pairs survive.
        junction_probs = out_jxn[0].sigmoid().cpu().numpy()
        donors, acceptors = (junction_probs >= CANDIDATE_SITE_THRESHOLD).nonzero()
        prediction["junctions"] = [
            {"donor": int(sites[d]) + 1, "acceptor": int(sites[a]) + 1, "probability": float(junction_probs[d, a])}
            for d, a in zip(donors, acceptors, strict=True)
        ]

        masks = decode_topk_tx(out_tx[0].cpu().numpy(), sites, n, num_candidates, k=num_transcripts)
        transcripts: list[list[tuple[int, int]]] = []
        for mask in masks:
            exons = _exon_intervals(mask)
            if exons not in transcripts:
                transcripts.append(exons)
        prediction["transcripts"] = [{"exons": exons} for exons in transcripts]
        return prediction

    def predict(
        self,
        sequences: list[str],
        assembly: str,
        num_transcripts: int,
        device: str,
    ) -> list[dict[str, Any]]:
        """Predict splice sites, junctions, and transcripts for each sequence."""
        self._ensure_loaded(device)
        results = []
        for i, sequence in enumerate(sequences):
            logger.update_status(f"Predicting sequence {i + 1}/{len(sequences)}")
            results.append(self._predict_one(sequence, assembly, num_transcripts))
        return results


def _exon_intervals(mask: Any) -> list[tuple[int, int]]:
    """Convert a 0/1 exon mask into 1-based inclusive exon intervals."""
    import numpy as np

    edges = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s) + 1, int(e)) for s, e in zip(starts, ends, strict=True)]


# ============================================================================
# Dispatch
# ============================================================================
_model: SpliceAI2Model | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = SpliceAI2Model()

    operation = input_dict["operation"]
    if operation == "score":
        results = _model.score(
            variants=input_dict["variants"],
            reference_fasta=input_dict["reference_fasta"],
            assembly=input_dict["assembly"],
            device=input_dict["device"],
        )
        return {"results": results}
    if operation == "predict":
        results = _model.predict(
            sequences=input_dict["sequences"],
            assembly=input_dict["assembly"],
            num_transcripts=input_dict["num_transcripts"],
            device=input_dict["device"],
        )
        return {"results": results}
    raise ValueError(f"spliceai2: unknown operation {operation!r}; valid: ['predict', 'score']")


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
    device = _model.device if _model and _model.device else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("spliceai2: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
