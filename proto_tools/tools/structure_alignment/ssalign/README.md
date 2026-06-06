<a href="https://bio-pro.mintlify.app/tools/structure-alignment/ssalign"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# SSAlign

> [!NOTE]
> **License:** SSAlign uses Custom (upstream repo unlicensed) for code and MIT for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/ISYSLAB-HUST/SSAlign) and [model weights license](https://huggingface.co/westlake-repl/SaProt_650M_AF2) for full terms.
>
> The upstream SSAlign repository ships no license file, so this toolkit reimplements the published method from permissively-licensed components rather than redistributing upstream code.

## Overview

[SSAlign](https://github.com/ISYSLAB-HUST/SSAlign) is an ultrafast, sensitive protein structure search method from the [ISYSLAB](http://bioinfo.isyslab.info/ssalign/) group at Huazhong University of Science and Technology. It encodes each structure with a structure-aware protein language model and runs a two-stage retrieval-then-refine search, reaching TM-align-comparable accuracy at exceptional speed on structure databases of unprecedented scale. It is the appropriate tool for screening a query fold against millions of predicted structures where pairwise aligners become a bottleneck.

## Background

SSAlign ([Wang et al., 2025](https://doi.org/10.1101/2025.07.03.662911)) frames structure search as embedding retrieval followed by selective alignment refinement. Each structure is first reduced to a [3Di structural-alphabet sequence](https://doi.org/10.1038/s41587-023-01773-0), which captures the tertiary contacts of every residue as a discrete structural letter. The 3Di letters are interleaved with the amino-acid sequence and passed through [SaProt-650M](https://huggingface.co/westlake-repl/SaProt_650M_AF2) ([Su et al., 2024](https://openreview.net/forum?id=6MRm3G4NiU)), a structure-aware ESM2-style protein language model whose vocabulary jointly tokenizes residue identity and local structure. Mean-pooling the per-residue hidden states yields a single 1280-dimensional structure embedding per protein.

The two-stage search then trades exhaustive alignment for fast vector retrieval. An Entropy Reduction Module (ERM) whitens and dimensionality-reduces the embeddings, and a [FAISS](https://github.com/facebookresearch/faiss) cosine index prefilters the database to a small candidate set per query. Borderline candidates whose cosine similarity falls below a threshold are then re-ranked by SAligner, a Needleman–Wunsch–Gotoh global aligner that scores candidates by 3Di global-alignment score, recovering sensitivity that pure embedding cosine would miss. The authors report that on AFDB50 this pipeline is roughly two orders of magnitude faster than exhaustive structural alignment, while improving sensitivity on SCOPe40 (reported relative AUC gains of +20.2% at the family level and +33.3% at the superfamily level) and retrieving more high-quality Swiss-Prot matches.

This toolkit reimplements the published SSAlign method using permissively-licensed components — SaProt weights (MIT), and `affine-gaps`, FAISS, and `transformers` from pip — and does not redistribute upstream code. Targets can either be supplied directly and embedded and indexed on the fly, or read from a prebuilt SSAlignDB directory for repeated large-scale search.

### Learning Resources

- [ISYSLAB-HUST/SSAlign](https://github.com/ISYSLAB-HUST/SSAlign) (ISYSLAB, Huazhong University of Science and Technology). The original SSAlign repository and reference implementation.
- [SSAlign web server](http://bioinfo.isyslab.info/ssalign/) (ISYSLAB). The authors' hosted search service against precomputed structure databases.

## Tools

### SSAlign Structure Search (`ssalign-search`)

Searches one or more query structures against a structure database and returns a ranked hit-bundle per query. Each query is embedded with 3Di plus SaProt, prefiltered with a FAISS cosine index (whitened in mode 2), and (in refine mode) re-ranked with SAligner 3Di global alignment.

#### Applications

This tool is the structural analogue of a large-scale homology search and is the appropriate first step for screening a query fold against millions of predicted structures, mining the [AlphaFold Database](https://alphafold.ebi.ac.uk/) for distant structural relatives that fall below the sequence twilight zone, and assessing whether a designed protein recapitulates a known fold at a scale where pairwise aligners become a runtime bottleneck.

#### Usage Tips

- **Provide exactly one target source: `target_structures` (mode 1) or `ssalign_db` (mode 2), never both.** `target_structures` embeds and indexes a target set in memory at call time (build on the fly), while `ssalign_db` searches a prebuilt SSAlignDB directory. Supplying both, or neither, fails input validation.
- **`mode` selects how far the search runs.** `mode=0` returns the cosine prefilter ranking only, while the default `mode=1` additionally re-ranks below-threshold candidates with SAligner 3Di global alignment to recover sensitivity. Use `mode=0` for the fastest coarse pass and `mode=1` for the published accuracy.
- **`prefilter_target`, `prefilter_threshold`, and `max_target` control the candidate funnel.** `prefilter_target` (default 2000) is the top-K cosine candidates retained per query; candidates with cosine below `prefilter_threshold` (default 0.3) are the ones SAligner refines in mode 1; `max_target` (default 1000) caps the final hits returned and must be `<= prefilter_target`.
- **Mode 1 ranks by raw SaProt cosine; mode 2 applies the database's trained ERM whitening.** SSAlign's entropy-reduction (whitening) transform is fit on a full database, so on-the-fly search over a user-supplied target set ranks by raw SaProt embedding cosine, while a prebuilt `ssalign_db` applies its shipped `mu`/`W`. `dim` selects which prebuilt index to search (mode 2) and must match a shipped index dimension; it is ignored when building on the fly.
- **SaProt-650M (~2.5 GB) auto-downloads from HuggingFace on first run, and a GPU is strongly recommended.** The first call fetches the weights to the standalone environment's cache; subsequent calls reuse them. SaProt embedding dominates runtime, so a CUDA GPU is strongly recommended for any non-trivial target set.
- **Mode 2 needs a manually-provisioned SSAlignDB directory.** No database is bundled. The `ssalign_db` directory must contain the FAISS index (`*_IndexFlatIP_{dim}_faiss.index`), the whitening `mu`/`W` arrays, and the id/sequence `npz`; a missing or incomplete directory fails fast with a `MissingAssetError`.
- **Using it as a structural-novelty check.** `ss_score` is a predicted average TM-score, computed as `0.55*prefilter_score + 0.56` (a generic SSAlign linear fit of the embedding cosine onto `(TM-Score1+TM-Score2)/2`). It is therefore **monotonic in `prefilter_score`** and carries no information beyond the cosine; higher means closer to a known fold. Because of the `0.56` intercept it does not span `[0, 1]` but floors near `0.56` for any non-negative cosine, so a nearest hit almost never falls below it. **Do not turn a TM cutoff into an `ss_score` threshold** (e.g. `< 0.4` sits below the floor and would reject every candidate). Instead **rank by `ss_score` (equivalently `prefilter_score`) ascending**, where lower means more novel, and treat a top hit approaching `~1.1` as clearly similar. For an exact TM, align the design to the top hits with `usalign`/`tmalign` (the database stores embeddings, not coordinates). Search is **per chain**, so score a multi-chain complex one chain at a time.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every SSAlign tool in this toolkit (`ssalign-search`).

- **Runs in an isolated standalone environment.** The toolkit builds and manages its own Python environment (PyTorch, `transformers`, `faiss-cpu`, `affine-gaps`); no manual environment setup is required, and the heavy SaProt dependency stays out of the core install.
- **3Di extraction is foldseek-free.** 3Di tokens are produced by [`mini3di`](https://github.com/althonos/mini3di), a pure-Python NumPy port of foldseek's 3Di VQ-VAE encoder that uses foldseek's trained weights, so no foldseek binary is downloaded or run. This makes the tool usable where the foldseek binary is unavailable or disallowed; the 3Di alphabet and the SAligner substitution matrix remain foldseek's.
  - **Not bit-identical to the foldseek binary.** mini3di shares foldseek's weights and 3Di alphabet, so agreement is high on well-formed cores, but it is a reimplementation: NumPy floating-point order, terminal/edge handling, and unencodable residues (mini3di assigns these the `D` state rather than foldseek's masked `X`) can differ slightly at structure margins, as can Bio.PDB parsing vs foldseek's internal reader. Treat the 3Di (and the resulting embedding search) as foldseek-equivalent, not byte-equal.
- **SaProt-650M weights are MIT and download on first run.** The ~2.5 GB `westlake-repl/SaProt_650M_AF2` weights are fetched from HuggingFace and cached for reuse. SaProt embedding is the dominant cost, so a CUDA GPU is strongly recommended.
- **The upstream SSAlign repository is unlicensed.** This toolkit reimplements the published method from permissively-licensed components and does not redistribute upstream code; consult the authors before reusing the original SSAlign source (see the License callout above).
