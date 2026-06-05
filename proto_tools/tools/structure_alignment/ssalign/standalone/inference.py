"""SSAlign standalone runner: SaProt embedding + FAISS cosine prefilter + SAligner 3Di refine."""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from standalone_helpers import (
    get_logger,
    get_pytorch_memory_stats,
    serialize_output,
    set_torch_seed,
)

# torch and faiss-cpu each bundle their own OpenMP runtime; on macOS loading both aborts
# the process ("OMP: Error #15: libomp.dylib already initialized"). Allow the duplicate
# (the standard, cross-platform-safe workaround) before either is imported lazily below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = get_logger(__name__)

# ── 3Di alphabet + foldseek substitution matrix ─────────────────────────────
# 21 letters; 'X' is the unknown/masked 3Di state. Rows/cols of _MAT3DI follow
# this exact order. The 21x21 matrix is the canonical foldseek 3Di substitution
# matrix (last row/col for 'X' are zeros).
_3DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

_MAT3DI = np.array(
    [
        [6, -3, 1, 2, 3, -2, -2, -7, -3, -3, -10, -5, -1, 1, -4, -7, -5, -6, 0, -2, 0],
        [-3, 6, -2, -8, -5, -4, -4, -12, -13, 1, -14, 0, 0, 1, -1, 0, -8, 1, -7, -9, 0],
        [1, -2, 4, -3, 0, 1, 1, -3, -5, -4, -5, -2, 1, -1, -1, -4, -2, -3, -2, -2, 0],
        [2, -8, -3, 9, -2, -7, -4, -12, -10, -7, -17, -8, -6, -3, -8, -10, -10, -13, -6, -3, 0],
        [3, -5, 0, -2, 7, -3, -3, -5, 1, -3, -9, -5, -2, 2, -5, -8, -3, -7, 4, -4, 0],
        [-2, -4, 1, -7, -3, 6, 3, 0, -7, -7, -1, -2, -2, -4, 3, -3, 4, -6, -4, -2, 0],
        [-2, -4, 1, -4, -3, 3, 6, -4, -7, -6, -6, 0, -1, -3, 1, -3, -1, -5, -5, 3, 0],
        [-7, -12, -3, -12, -5, 0, -4, 8, -5, -11, 7, -7, -6, -6, -3, -9, 6, -12, -5, -8, 0],
        [-3, -13, -5, -10, 1, -7, -7, -5, 9, -11, -8, -12, -6, -5, -9, -14, -5, -15, 5, -8, 0],
        [-3, 1, -4, -7, -3, -7, -6, -11, -11, 6, -16, -3, -2, 2, -4, -4, -9, 0, -8, -9, 0],
        [-10, -14, -5, -17, -9, -1, -6, 7, -8, -16, 10, -9, -9, -10, -5, -10, 3, -16, -6, -9, 0],
        [-5, 0, -2, -8, -5, -2, 0, -7, -12, -3, -9, 7, 0, -2, 2, 3, -4, 0, -8, -5, 0],
        [-1, 0, 1, -6, -2, -2, -1, -6, -6, -2, -9, 0, 4, 0, 0, -2, -4, 0, -4, -5, 0],
        [1, 1, -1, -3, 2, -4, -3, -6, -5, 2, -10, -2, 0, 5, -2, -4, -5, -1, -2, -5, 0],
        [-4, -1, -1, -8, -5, 3, 1, -3, -9, -4, -5, 2, 0, -2, 6, 2, 0, -1, -6, -3, 0],
        [-7, 0, -4, -10, -8, -3, -3, -9, -14, -4, -10, 3, -2, -4, 2, 6, -6, 0, -11, -9, 0],
        [-5, -8, -2, -10, -3, 4, -1, 6, -5, -9, 3, -4, -4, -5, 0, -6, 8, -9, -5, -5, 0],
        [-6, 1, -3, -13, -7, -6, -5, -12, -15, 0, -16, 0, 0, -1, -1, 0, -9, 3, -10, -11, 0],
        [0, -7, -2, -6, 4, -4, -5, -5, 5, -8, -6, -8, -4, -2, -6, -11, -5, -10, 8, -6, 0],
        [-2, -9, -2, -3, -4, -2, 3, -8, -8, -9, -9, -5, -5, -5, -3, -9, -5, -11, -6, 9, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)

_GAP_OPEN = -10
_GAP_EXTEND = -1

# SaProt structure-aware PLM. SSAlign uses the AF2-trained 650M variant (matching the AlphaFold
# structures its prebuilt databases were built from); upstream hardcodes this single model.
_SAPROT_REPO = "westlake-repl/SaProt_650M_AF2"


# ============================================================================
# SaProt embedder (lazy, persistent)
# ============================================================================
class SaProtEmbedder:
    """Lazy SaProt structure-aware PLM wrapper that mean-pools last-layer states."""

    def __init__(self) -> None:
        """Set up empty state; weights load lazily on first embed."""
        self._loaded = False
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None

    def load(self, device: str) -> None:
        """Load SaProt tokenizer + masked-LM weights from HuggingFace onto device."""
        from transformers import EsmForMaskedLM, EsmTokenizer

        logger.update_status(f"Loading SaProt ({_SAPROT_REPO}) on {device}")
        self.tokenizer = EsmTokenizer.from_pretrained(_SAPROT_REPO)
        self.model = EsmForMaskedLM.from_pretrained(_SAPROT_REPO).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self._loaded = True

    def to_device(self, device: str) -> None:
        """Move the loaded model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("ssalign: cannot move unloaded SaProt model — call load() first")
        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def embed(self, combined_seqs: list[str], batch_size: int, device: str) -> np.ndarray:
        """Mean-pool last-layer SaProt embeddings over residue tokens (excluding CLS/EOS).

        ``combined_seqs`` are interleaved "AA+3di" strings (one 2-char token per
        residue, e.g. "Md" for residue M with 3Di state D). Returns a
        ``(N, hidden)`` float32 array, one pooled vector per input sequence.
        """
        import torch

        if not self._loaded:
            self.load(device)
        elif self.device != device:
            self.to_device(device)

        all_pooled: list[np.ndarray] = []
        for start in range(0, len(combined_seqs), batch_size):
            batch = combined_seqs[start : start + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[-1]  # (B, L, H)

            # Mean-pool over residue tokens: attention_mask with CLS (index 0) and the
            # per-row EOS (last non-pad position) zeroed out.
            attn = enc["attention_mask"].clone()
            attn[:, 0] = 0
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            attn[torch.arange(attn.size(0), device=attn.device), last_idx] = 0

            mask = attn.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_pooled.append(pooled.float().cpu().numpy())

        return np.concatenate(all_pooled, axis=0)


# ============================================================================
# 3Di extraction (mini3di: pure-Python NumPy port of foldseek's 3Di VQ-VAE encoder)
# ============================================================================
def _structures_to_3di(items: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Extract (aa_seq, 3Di_seq) per structure with mini3di, first protein chain only.

    mini3di is a pure-Python (NumPy) port of foldseek's 3Di VQ-VAE encoder using
    foldseek's trained weights, so it produces foldseek-compatible 3Di without the
    foldseek binary. ``items`` are ``[{"id", "text", "format"}]`` in caller order;
    returns ``(aa_seq, 3Di_seq)`` per input in that order, using the first chain that
    has amino-acid residues (skipping nucleic-acid/ligand-only chains). Modified
    residues (e.g. MSE) map to their canonical AA via biopython's extended table;
    unknowns map to 'X' (SaProt-masked downstream by _build_combined_seq).
    """
    import io

    import mini3di
    from Bio.Data.PDBData import protein_letters_3to1_extended
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    encoder = mini3di.Encoder()
    pdb_parser = PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
    cif_parser = MMCIFParser(QUIET=True)  # type: ignore[no-untyped-call]

    results: list[tuple[str, str]] = []
    for item in items:
        parser = cif_parser if item["format"].lower() in {"cif", "mmcif"} else pdb_parser
        structure = parser.get_structure(item["id"], io.StringIO(item["text"]))  # type: ignore[no-untyped-call]
        try:
            model = next(structure.get_models())
        except StopIteration as exc:
            raise RuntimeError(f"ssalign: no model found in structure {item['id']}.") from exc

        # Use the first chain that yields amino-acid residues (foldseek emitted 3Di for
        # the first protein chain, so a leading nucleic-acid/ligand chain must not abort).
        for chain in model.get_chains():
            n, ca, c, cb, aa = [], [], [], [], []
            for res in chain:
                if not is_aa(res, standard=False):  # type: ignore[no-untyped-call]
                    continue
                coords = {atom.get_name(): atom.get_coord() for atom in res}
                if not {"N", "CA", "C"} <= coords.keys():
                    continue
                n.append(coords["N"])
                ca.append(coords["CA"])
                c.append(coords["C"])
                cb.append(coords.get("CB", (math.nan, math.nan, math.nan)))
                aa.append(protein_letters_3to1_extended.get(res.get_resname().strip().upper(), "X"))
            if ca:
                break
        else:
            raise RuntimeError(
                f"ssalign: structure {item['id']} has no protein chain with N/CA/C backbone "
                "atoms; it may be empty, non-protein, or malformed."
            )
        states = encoder.encode_atoms(ca=np.asarray(ca), cb=np.asarray(cb), n=np.asarray(n), c=np.asarray(c))
        results.append(("".join(aa), encoder.build_sequence(states).upper()))
    return results


def _build_combined_seq(aa: str, threeDi: str) -> str:
    """Interleave AA (upper) + 3Di (lower) per residue; unknown AA/3Di -> '#' (SaProt mask)."""
    combined = "".join(a + t.lower() for a, t in zip(aa, threeDi, strict=True))
    return combined.replace("X", "#").replace("x", "#")


def _threeDi_only(threeDi: str) -> str:
    """Return the uppercase 3Di string (mini3di output is uppercased at extraction)."""
    return threeDi.upper()


def _threeDi_from_combined(combined: str) -> str:
    """Recover the uppercase target 3Di string from a stored combined seq (mode 2).

    Takes the 2nd char of each 2-char token; '#' (mask) maps back to 'X'.
    """
    threeDi = combined[1::2].upper()
    return threeDi.replace("#", "X")


# ============================================================================
# Whitening (mode 2: apply the prebuilt DB's shipped mu/W) + FAISS index
# ============================================================================
def _apply_whitening(emb: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Whiten ``emb`` with (mu, W) and L2-normalize the result for cosine search."""
    import faiss

    whitened: np.ndarray = ((emb - mu) @ w).astype(np.float32)
    faiss.normalize_L2(whitened)
    return whitened


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 copy of ``vecs`` with each row L2-normalized."""
    import faiss

    v = np.ascontiguousarray(vecs.astype("float32"))
    faiss.normalize_L2(v)
    return v


def _l2_index(vecs: np.ndarray) -> tuple[Any, np.ndarray]:
    """Build an L2-normalized inner-product (cosine) FAISS index over ``vecs``."""
    import faiss

    v = _l2_normalize(vecs)
    idx = faiss.IndexFlatIP(v.shape[1])
    idx.add(v)
    return idx, v


# ============================================================================
# SAligner (affine-gaps) 3Di refine adapter
# ============================================================================
def _saligner_score(q_3di: str, t_3di: str) -> float:
    """SAligner 3Di global-alignment score (affine-gaps NW-Gotoh, foldseek mat3di, gap -10/-1)."""
    from affine_gaps import needleman_wunsch_gotoh_score

    return float(
        needleman_wunsch_gotoh_score(
            q_3di,
            t_3di,
            substitution_alphabet=_3DI_ALPHABET,
            substitution_matrix=_MAT3DI,
            gap_opening=_GAP_OPEN,
            gap_extension=_GAP_EXTEND,
        )
    )


# ============================================================================
# Prebuilt SSAlignDB loading (mode 2)
# ============================================================================
def _load_ssalign_db(db_dir: str, dim: int) -> tuple[Any, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load a prebuilt SSAlignDB (FAISS index + whitening mu/W + id/seq npz) -> (index, mu, W[:, :dim], ids, seqs).

    The tool layer's ``_require_ssalign_db`` is the user-facing (skippable) check before dispatch;
    this is a backstop that raises if the directory is missing or incomplete.
    """
    path = Path(db_dir)
    idx_files = sorted(path.glob(f"*_IndexFlatIP_{dim}_faiss.index")) if path.is_dir() else []
    mu_files = sorted(path.glob("*_whitening_mu.npy")) if path.is_dir() else []
    w_files = sorted(path.glob("*_whitening_W.npy")) if path.is_dir() else []
    seq_files = sorted(path.glob("*_id_Seq.npz")) if path.is_dir() else []

    if not (path.is_dir() and idx_files and mu_files and w_files and seq_files):
        raise FileNotFoundError(
            f"ssalign: SSAlignDB at {db_dir!r} is missing or incomplete for dim={dim} "
            f"(need *_IndexFlatIP_{dim}_faiss.index, *_whitening_mu.npy, *_whitening_W.npy, *_id_Seq.npz)."
        )

    import faiss

    mu = np.load(mu_files[0])
    w = np.load(w_files[0])[:, :dim]
    npz = np.load(seq_files[0], allow_pickle=True)
    ids = [str(i) for i in npz["ids"]]
    seqs = [str(s) for s in npz["seqs"]]
    index = faiss.read_index(str(idx_files[0]))
    return index, mu, w, ids, seqs


# ============================================================================
# Dispatch
# ============================================================================
_saprot: SaProtEmbedder | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run an SSAlign search: build/load the index once, then search every query.

    The framework does not unroll the query list, so the (possibly expensive)
    target index is built or loaded a single time and reused across all queries.
    """
    global _saprot

    if _saprot is None:
        _saprot = SaProtEmbedder()

    set_torch_seed(input_dict.get("seed"))

    device = input_dict["device"]
    dim = input_dict["dim"]
    threads = input_dict["num_threads"]
    batch_size = input_dict["batch_size"]
    mode = input_dict["mode"]

    # ── Embed queries (3Di + SaProt), aligned with input order ───────────────
    query_items = input_dict["queries"]
    q_ids = [item["id"] for item in query_items]
    query_3di = _structures_to_3di(query_items)  # [(aa, 3Di)] in input order
    q_combined = [_build_combined_seq(aa, threeDi) for aa, threeDi in query_3di]
    q_3di = [_threeDi_only(threeDi) for _, threeDi in query_3di]
    q_emb = _saprot.embed(q_combined, batch_size, device)

    # ── Build / load the target index ────────────────────────────────────────
    if input_dict["target_structures"] is not None:
        # MODE 1: embed + index targets on the fly.
        target_items = input_dict["target_structures"]
        t_ids = [item["id"] for item in target_items]
        target_3di = _structures_to_3di(target_items)
        t_combined = [_build_combined_seq(aa, threeDi) for aa, threeDi in target_3di]
        t_3di = {tid: _threeDi_only(threeDi) for tid, (_, threeDi) in zip(t_ids, target_3di, strict=True)}
        t_emb = _saprot.embed(t_combined, batch_size, device)

        # On-the-fly: raw L2-normalized SaProt cosine over the full embedding dim. (ERM whitening
        # is fit on a large database; that is the prebuilt SSAlignDB used in mode 2, below.)
        index, _ = _l2_index(t_emb)
        q_vec = _l2_normalize(q_emb)
    else:
        # MODE 2: load a prebuilt SSAlignDB and apply its shipped whitening.
        index, mu, w, t_ids, t_seqs = _load_ssalign_db(input_dict["ssalign_db"], dim)
        t_3di = {tid: _threeDi_from_combined(seq) for tid, seq in zip(t_ids, t_seqs, strict=True)}
        q_vec = _apply_whitening(q_emb, mu, w)

    # ── Cosine prefilter ─────────────────────────────────────────────────────
    import faiss

    faiss.omp_set_num_threads(threads)
    k = min(input_dict["prefilter_target"], len(t_ids))
    distances, indices = index.search(np.ascontiguousarray(q_vec), k)

    prefilter_threshold = input_dict["prefilter_threshold"]
    max_target = input_dict["max_target"]

    results: list[dict[str, Any]] = []
    for qi, qid in enumerate(q_ids):
        candidates = []
        for j in range(len(indices[qi])):
            ti = indices[qi][j]
            if ti < 0:
                continue
            # IndexFlatIP self-similarity can exceed 1 by tiny FP error; clamp to the cosine domain.
            cos = max(-1.0, min(1.0, float(distances[qi][j])))
            candidates.append((t_ids[ti], cos))
        hits = _rank_hits(candidates, mode, prefilter_threshold, q_3di[qi], t_3di, max_target)
        results.append({"query_id": qid, "hits": hits})

    output: dict[str, Any] = serialize_output({"results": results})
    return output


def _rank_hits(
    candidates: list[tuple[str, float]],
    mode: int,
    prefilter_threshold: float,
    query_3di: str,
    t_3di: dict[str, str],
    max_target: int,
) -> list[dict[str, Any]]:
    """Rank prefilter candidates into the final per-query hit list.

    Mode 0: cosine-only, sorted by cosine desc (no SAligner refine).
    Mode 1: above-threshold hits (by cosine) rank first, unrefined; below-threshold
    hits are SAligner-refined and ranked after, by saligner score desc. Truncated
    to ``max_target`` with 1-indexed ranks.
    """
    hits: list[dict[str, Any]]
    if mode == 0:
        ordered = sorted(candidates, key=lambda c: c[1], reverse=True)
        hits = [
            {"target_id": tid, "prefilter_score": cos, "saligner_score": None, "refined": False} for tid, cos in ordered
        ]
    else:
        above = [(tid, cos) for tid, cos in candidates if cos >= prefilter_threshold]
        below = [(tid, cos) for tid, cos in candidates if cos < prefilter_threshold]
        above.sort(key=lambda c: c[1], reverse=True)

        # SAligner (affine-gaps NW-Gotoh) returns a finite int score for each below-threshold hit.
        refined = [(tid, cos, _saligner_score(query_3di, t_3di[tid])) for tid, cos in below]
        refined.sort(key=lambda c: c[2], reverse=True)

        hits = [
            {"target_id": tid, "prefilter_score": cos, "saligner_score": None, "refined": False} for tid, cos in above
        ]
        hits += [
            {"target_id": tid, "prefilter_score": cos, "saligner_score": sal, "refined": True}
            for tid, cos, sal in refined
        ]

    hits = hits[:max_target]
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
        hit["ss_score"] = 0.55 * hit["prefilter_score"] + 0.56
    return hits


# ============================================================================
# Device-manager protocol functions
# ============================================================================
def to_device(device: str) -> dict[str, Any]:
    """Move the SaProt model to ``device`` (called by DeviceManager)."""
    global _saprot
    if _saprot is not None and _saprot._loaded:
        _saprot.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage (called by DeviceManager for monitoring)."""
    global _saprot
    device = _saprot.device if _saprot and _saprot._loaded and _saprot.device else 0
    stats: dict[str, Any] = get_pytorch_memory_stats(device)
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("ssalign: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
