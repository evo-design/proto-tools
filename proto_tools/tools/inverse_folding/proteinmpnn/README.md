<a href="https://bio-pro.mintlify.app/tools/inverse-folding/proteinmpnn"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ProteinMPNN

![ProteinMPNN](https://proto-bio.github.io/proto-assets/images/tool/proteinmpnn/hero.png)

> [!NOTE]
> **License:** ProteinMPNN is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE) for full terms.

## Overview

First released in 2022 by the [Baker Lab at the Institute for Protein Design](https://www.ipd.uw.edu/), Protein Message Passing Neural Network (ProteinMPNN) is a deep-learning model for inverse folding, predicting which sequences fold into a given 3D backbone. It has become a standard sequence-design step in de novo protein design, sharply outperforming prior physics-based methods in both accuracy and speed. It can design sequences for a target backbone, score how well a sequence fits a structure, and act as a differentiable structure-conditioned objective for gradient-based design.

## Background

ProteinMPNN ([Dauparas et al., 2022](https://doi.org/10.1126/science.add2187)) solves the inverse-folding problem: given a fixed protein backbone (the 3D coordinates of its N, C-alpha, C, and O atoms), predict an amino-acid sequence that will fold into that structure. It is the inverse of structure prediction and a core step in protein design, where a backbone is proposed first and a sequence that encodes it is designed afterwards.

Internally, ProteinMPNN encodes the backbone as a graph: each residue is a node connected to its 48 nearest neighbors in space, with edges featurized by inter-atomic distances between the backbone atoms (including a virtual C-beta). A neural network called a "message-passing" encoder turns this geometry into node and edge representations, and a decoder then generates the sequence autoregressively. ProteinMPNN is trained with a random decoding order rather than a fixed N-to-C order, so at inference any order can be used and arbitrary subsets of positions can be held fixed while the rest are designed in full structural context. It was trained on protein structures from the [Protein Data Bank](https://www.rcsb.org/). During training, a small amount of Gaussian noise was added to the backbone coordinates so the model is robust to imperfect, non-crystal backbones; this slightly lowers native-sequence recovery but yields sequences that more reliably fold to the intended structure. On native backbones it recovers roughly 52% of the native sequence on average, compared with roughly 33% for physically based Rosetta design. ProteinMPNN designs have been experimentally validated by X-ray crystallography and cryo-electron microscopy, and ProteinMPNN rescued monomers, cyclic homo-oligomers, nanoparticles, and target-binding proteins that had failed when designed with Rosetta or AlphaFold.

### Learning Resources

- [Sequence Design with ProteinMPNN](https://www.youtube.com/watch?v=zbpWFKjiXEk) - a video walkthrough of using ProteinMPNN for fixed-backbone protein sequence design.
- [MPNN - ML for protein sequence design](https://www.youtube.com/watch?v=6z4XmUAwdNA) - a talk on the message-passing machine-learning approach behind ProteinMPNN.

## Tools

### ProteinMPNN Sampling (`proteinmpnn-sample`)

Designs new sequences for a given backbone. Each input structure is encoded once and decoded into one or more candidate sequences, each returned with a perplexity and the sequence recovery against the structure's original sequence.

#### Applications

Use this to redesign or stabilize a natural protein, or to generate sequences for a de novo backbone (for example one from RFdiffusion). The standard design loop is to sample many sequences per backbone, rank by perplexity, and validate the top candidates with a structure predictor.

#### Usage Tips

- **`temperature` (default `0.1`) controls diversity.** Lower values are greedier and stay close to the single most likely sequence, while higher values sample more varied sequences. A value near `0.0` behaves like an argmax, and the temperature must be at least `0`.
- **Lower `batch_size` if you hit GPU memory limits.** It defaults to `num_sequences_per_structure`, so every requested sequence is generated in one forward pass. For large requests or long backbones this can exhaust GPU memory, and a smaller `batch_size` trades speed for lower memory.
- **`model_choice` selects the weights.** The default `proteinmpnn` is `v_48_020`. The `v_48_002`, `v_48_010`, and `v_48_030` variants are trained with increasing backbone noise, which makes designs more robust and diverse at the cost of native-sequence recovery. `abmpnn` is antibody-tuned. Use `soluble` when the design must be water-soluble, because the default model tends to place hydrophobic residues on membrane-like surfaces whereas `soluble` is retrained with transmembrane proteins excluded.
- **`fixed_positions` is counted from 1, not 0.** Listing a position keeps that residue at its input identity, which is how you preserve catalytic or interface residues while redesigning everything else.
- **`excluded_amino_acids` forbids residue types everywhere.** Use it to keep unwanted residues out of every design, for example `["C"]` to avoid introducing cysteines.
- **`backbone_noise` (default `0.0`) and `seed`.** `backbone_noise` adds Gaussian noise in angstroms to the input backbone. Small values such as `0.02` increase diversity at some cost in recovery. Set `seed` for reproducible sampling.
- **Designing a symmetric homooligomer? Use `proteinmpnn-sample-homooligomer` instead.** This tool designs each chain independently against placeholder partner sequences. When the chains you select are sequence-identical it logs a one-line nudge toward the dedicated tool below.

### ProteinMPNN Homooligomer Sampling (`proteinmpnn-sample-homooligomer`)

Designs a symmetric homooligomer by tying corresponding positions across the protomer chains during decoding. A single protomer sequence is sampled with every tied protomer in context, then returned on all of them, so the interface residues are chosen against the partners they actually pack against.

#### Applications

Use this for any C-symmetric assembly — homodimers, homotrimers, designed cages and fibers — where every protomer must carry the same sequence. The alternative, designing one protomer against placeholder partners and copying the result onto the others, picks a sequence the model never saw inside its own interface.

#### Usage Tips

- **Selections are per protomer, not per chain.** `protomer_chains` lists the chains to tie (at least two, all the same length) and `protomer_fixed_positions` is a flat list of 1-indexed positions *within one protomer*, applied to all of them. There is deliberately no per-chain `fixed_positions`: because tied decoding averages the per-position biases across the tied group, a position fixed on only some protomers would silently fix it on all of them, and conflicting per-chain choices would resolve arbitrarily.
- **`temperature` is not comparable to `proteinmpnn-sample`.** Tying averages the per-protomer logits together with their Gumbel noise, so effective sampling noise falls by roughly `1/sqrt(n)` for an `n`-mer. An `n`-mer at a given temperature is less diverse than a monomer at the same setting; raise `temperature` to compensate.
- **Equal length is not the same as homooligomer.** Protomers that match in length but differ in sequence are accepted as a pseudo-symmetric assembly, with a warning, since one sequence will be imposed on all of them. If you meant to design those chains independently, use `proteinmpnn-sample`.
- **Only the chains you list reach the model.** Chains outside `protomer_chains` are copied into the output unchanged with `redesigned=False`, but they do *not* condition the design — ProteinMPNN is given only the selected chains, so a partner or scaffold chain you leave out is invisible to it. List every chain that should inform the design. (This matches `proteinmpnn-sample`, where unselected chains are likewise excluded from the model.)
- **Protomers must share residue numbering.** Staging renumbers each chain 1..N over its *observed* residues, so two protomers whose unresolved gaps fall in different places would tie position `i` to different structural residues. Mismatched numbering is rejected rather than warned about. Protomers numbered in separate ranges (`1..`, `101..`) are fine; it is the pattern of gaps that must agree.

### ProteinMPNN Scoring (`proteinmpnn-score`)

Evaluates how well existing sequences fit a structure. Each (sequence, structure) pair is scored under ProteinMPNN's structure-conditioned likelihood, returning log-likelihood, average log-likelihood, and perplexity, with optional per-position logits.

#### Applications

Use this to rank candidate sequences or point mutations by structural compatibility without generating new ones: compare designs, assess the effect of a substitution, or filter a library before experimental testing. Lower perplexity indicates a better structure-sequence fit.

#### Usage Tips

- **Set `excluded_positions` per (sequence, structure) pair to score only part of a chain.** It lives on each input pair as a `{chain: [positions]}` selection, not in the config. Listed positions are left out of the log-likelihood and perplexity, so the score reflects just the residues you care about; the residues still condition the model. Scoring uses `excluded_positions` while the sampling tools use `fixed_positions`, which holds an identity rather than omitting a position from a metric. **NOTE:** Positions are per chain and counted from 1, not 0, to match biological residue selection conventions.
- **Residues ProteinMPNN's parser cannot read are left out of the score automatically.** GFP's CRO chromophore is one; a warning names the chain and position, and those residues never contribute to the reported metrics. Asking for `return_logits` on such a structure is rejected, since the returned rows could not be aligned to your sequence.
- **`return_logits` (default `False`) has a size trade-off.** Enabling it returns a per-position `(sequence length x 20)` logit array per sequence, in the shared amino-acid order `ACDEFGHIKLMNPQRSTVWY` reported as `vocab`. The model scores in its own order and over an extra unknown-residue column; both are handled internally, so a softmax over the returned columns is not the model's full distribution. That array dominates output size and memory for long sequences or large batches, so leave it off unless you need it.

### ProteinMPNN Gradient (`proteinmpnn-gradient`)

Exposes ProteinMPNN as a differentiable structure-conditioned objective: given a relaxed `(L, 20)` sequence distribution and a backbone, it returns the mean negative log-likelihood and its gradient with respect to the input logits, for use as a loss in gradient-based or MCMC sequence optimization.

#### Applications

Use this when ProteinMPNN is one term in a larger optimization over a continuous sequence representation (for example combined with other structure or property objectives), rather than for standalone sampling. Set `compute_gradient=False` for forward-only NLL scoring, such as ranking MCMC proposals.

#### Usage Tips

- **`logits` columns are in the shared amino-acid order `ACDEFGHIKLMNPQRSTVWY`**, which every gradient tool uses and which you can import as `proto_tools.utils.PROTEIN_AMINO_ACIDS`. Columns are read by position, so passing a different ordering silently produces the wrong gradient.
- **`temperature`** controls how the input logits are converted into amino acid probabilities using `softmax(logits / temperature)`. Lower temperatures make the distribution more concentrated around the most likely amino acid at each position, while higher temperatures produce more uniform distributions. The default is `1.0`, and the returned gradient is calculated with respect to the input logits.
- **Advanced option:** `temperature=None` skips softmax and treats the input as probabilities rather than logits. Most users should leave the default in place. In this case, each row must contain nonnegative values that sum to 1, and the returned gradient is calculated with respect to those probabilities. This is generally less convenient for gradient-based optimization because updating probabilities directly can cause them to become negative or no longer sum to 1. Optimizing logits avoids this issue, since softmax always produces a valid probability distribution. However, working directly with probabilities can be useful when using an optimizer designed for probability distributions or when exact zero probabilities are needed to exclude particular tokens.
- **`compute_gradient` (default `True`).** Returns the gradient of the mean negative log-likelihood with respect to `logits`. Set `False` for forward-only scoring (`loss` only, `gradient` is `None`), for example to cheaply rank MCMC proposals.
- **`soft` and `hard` control the relaxation.** The default (`soft=1.0`, `hard=0.0`) feeds pure softmax probabilities. Set `hard=1.0` for a straight-through estimator — the forward pass sees a hard one-hot while gradients still flow through the probabilities. Lowering `soft` blends the raw input matrix in, which is differentiable but outside the model's training distribution. The negative-log-likelihood labels are the decoded sequence at every setting, so they do not move with `soft` or `hard`.
- **`fixed_positions` is counted from 1 and is left out of the objective.** Positions you list are excluded from both the loss and its gradient, so set it to optimize only the residues you are designing.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ProteinMPNN tool in this toolkit (`proteinmpnn-sample`, `proteinmpnn-sample-homooligomer`, `proteinmpnn-score`, `proteinmpnn-gradient`).

- **GPU recommended; CPU works but is slower.** ProteinMPNN is a small model and runs on CPU, but a GPU is far faster when sampling or scoring many sequences. Model weights (a few hundred MB across variants) download automatically on first use.
- **Reproducibility.** `proteinmpnn-sample`, `proteinmpnn-sample-homooligomer` and `proteinmpnn-gradient` are stochastic; set `seed` for reproducible runs.
- **Multi-chain designs come back chain by chain.** A design spanning several chains is returned as a complex holding one entry per chain, read through `design.chains` or `design.chain_sequence_map()`, not as one delimited string.
