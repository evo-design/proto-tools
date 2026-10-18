# Gated model access

A few tools use gated models or software that require accepting a license or
terms-of-use before the weights or binary can be downloaded. The access flow
depends on how the upstream author distributes them.

| Model/Tool | Source | Access |
|------------|--------|--------|
| ESM3 | HuggingFace: [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | Accept EvolutionaryScale license, then authenticate with HF (see below) |
| AlphaGenome | HuggingFace: [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) | Accept Google DeepMind terms, then authenticate with HF (see below) |
| SpliceAI2 | HuggingFace: [illumina-ai/SpliceAI2](https://huggingface.co/illumina-ai/SpliceAI2) | Accept the Illumina SpliceAI2 Model Terms of Use (non-commercial organizations only) in the form at the top of the model card, then authenticate with HF (see below) |
| X3DNA | Gated software (used by `x3dna-fiber`): register free at [x3dna.org](https://x3dna.org/) | After registering, see [`x3dna/SETUP.md`](../proto_tools/tools/structure_prediction/x3dna/SETUP.md) to stage it into the cache (no environment variable needed). |
| netMHCpan | Gated software (used by `netmhcpan-peptides`, `netmhcpan-protein`): request the academic package at [NetMHCpan-4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) | Academic/non-profit only, one site, no redistribution; commercial use needs a separate license from <health-software@dtu.dk>. After the download link arrives, see [`netmhcpan/SETUP.md`](../proto_tools/tools/epitope_prediction/netmhcpan/SETUP.md) — the separate `data.tar.gz` is also required. |

## For HuggingFace-gated models

1. Create a [HuggingFace](https://huggingface.co) account
2. Visit each model page above and **accept the license/terms**
3. Install the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and log in:
   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   hf auth login
   ```
   Or set the token directly in your environment:
   ```bash
   export HF_TOKEN=hf_...
   ```
