# Environment Dev Notes: Platform Compatibility
This file contains notes on platform compatibility with our current `setup.sh` scripts and `env_manager.py` system.

## Platform: Chimera (NVIDIA H100 GPU)

### System Info
| Property | Value |
|----------|-------|
| **OS** | Ubuntu 22.04.3 LTS (Jammy Jellyfish) |
| **Kernel** | 5.15.0-164-generic x86_64 |
| **Architecture** | x86_64 |
| **RAM** | 1.0 TB |

### GPU Info
| Property | Value |
|----------|-------|
| **GPU** | NVIDIA H100 80GB HBM3 |
| **Compute Capability** | 9.0 (sm_90) |
| **VRAM** | 80 GB |
| **CUDA Toolkit** | 12.9 |
| **Driver** | 535.183.01 |

### Key Platform Constraints
- **x86_64**: Standard architecture with good package support
- **Compute Capability 9.0**: H100 architecture with sm_90 support
- **CUDA 12.9**: Very new CUDA version (May 2025 release)
- **PyTorch Compatibility**: Some packages have issues with newer CUDA 12.9, particularly:
  - torch 2.6.0+cu126 installation failures
  - Symbol resolution issues with libnvJitLink.so.12 (`__nvJitLinkGetErrorLogSize_12_9`)

### Venv Status
**Last Updated:** 2026-02-16

| Category | Tool | GPU Version | Status | Notes |
|----------|------|-------------|--------|-------|
| Causal Models | evo1 | 2.7.1+cu128 | Working | 29/30 tests pass (1 skipped: slow); evo-model 0.5, transformers 5.1.0, numpy 2.4.2 |
| Gene Annotation | blast | N/A (no GPU) | Working | 11/11 tests pass |
| Gene Annotation | crispr_tracr | N/A (no GPU) | Not Tested | Requires nested conda env (Python 3.8 + sklearn 0.22 + 20 bioinformatics tools) |
| Gene Annotation | minced | N/A (no GPU) | Not Tested | Java tool; setup.sh downloads JAR into venv bin/ |
| Gene Annotation | mmseqs | N/A (no GPU) | Working | 29/29 tests pass |
| Gene Annotation | pyhmmer | N/A (no GPU) | Working | 21/21 tests pass |
| Inverse Folding | ligandmpnn | latest+cu126 | Working | 2/2 tests pass |
| Inverse Folding | proteinmpnn | jax-cuda12 | Working | 11/11 tests pass |
| Language Models | esm2 | 2.6.0+cu126 | Working | 11/11 tests pass |
| Language Models | esm3 | N/A | Working | 12/12 tests pass |
| Language Models | progen2 | CPU only | Working | 30/30 tests pass |
| Language Models | evo2 | 2.6.0+cu126 | Working | 32/32 tests pass; micromamba CUDA toolkit + cuDNN; sitecustomize.py preloads CUDA libs |
| ORF Prediction | orfipy | N/A (no GPU) | Working | 19/19 tests pass |
| ORF Prediction | prodigal | N/A (no GPU) | Working | 29/29 tests pass |
| RNA Splicing | splice_transformer | latest+cu126 | Working | 2/2 tests pass |
| Sequence Alignment | run_colabfold_search | N/A (no GPU) | Working | 22/22 tests pass |
| Sequence Alignment | mafft | N/A (no GPU) | Working | 40/40 tests pass |
| Sequence Scoring | enformer | 2.6.0+cu126 | Working | 8/8 tests pass |
| Sequence Scoring | borzoi | 2.6.0+cu126 | Working | 14/14 tests pass |
| Sequence Scoring | segmasker | N/A (no GPU) | Not Tested | Requires NCBI BLAST+ segmasker binary |
| Sequence Scoring | alphagenome | N/A | Not Working | 0/8 pass; 8 failing (tested before fixes in [#8](https://github.com/google-deepmind/alphagenome_research/issues/8) and [#10](https://github.com/google-deepmind/alphagenome_research/issues/10)) |
| Structure Design | rfdiffusion3 | latest+cu126 | Working | 3/3 tests pass |
| Structure Dynamics | bioemu | latest+cu126 | Working | 13/13 tests pass |
| Structure Prediction | esmfold | 2.6.0+cu126 | Working | Works |
| Structure Prediction | boltz | 2.10.0+cu130 | Working | Works |
| Structure Prediction | chai | 2.6.0+cu126 | Working | Works |
| Structure Prediction | protenix | 2.7.1+cu128 | Working | 9/9 tests pass |
| Structure Prediction | structure_metrics | N/A (no GPU) | Not Tested | Pure Python (BioPython); no external deps |
| Structure Prediction | viennarna | N/A (no GPU) | Working | 6/6 tests pass |


## Platform: DGX Spark (NVIDIA GB10 GPU)

### System Info
| Property | Value |
|----------|-------|
| **OS** | Ubuntu 24.04.3 LTS (Noble Numbat) |
| **Kernel** | 6.11.0-1016-nvidia aarch64 |
| **Architecture** | aarch64 (ARM64) |
| **RAM** | 120 GB |

### GPU Info
| Property | Value |
|----------|-------|
| **GPU** | NVIDIA GB10 |
| **Compute Capability** | 12.1 (sm_121) |
| **VRAM** | 120 GB (Shared with system RAM) |
| **CUDA Toolkit** | 13.0 |
| **Driver** | 580.95.05 |

### Key Platform Constraints
- **aarch64**: Many pip packages only ship x86_64 pre-built wheels (flash-attn, mafft binaries, etc.)
- **Compute Capability 12.1**: Very new — older pinned torch versions (< 2.9) lack sm_121 support
- **CUDA 13.0**: Newest CUDA version — resolves to `cu130` or `cu128` builds

### Venv Status
**Last Updated:** 2026-02-12

| Category | Tool | GPU Version | Status | Notes |
|----------|------|-------------|--------|-------|
| Causal Models | evo1 | N/A | Not Tested | |
| Gene Annotation | blast | N/A (no GPU) | Working | Works |
| Gene Annotation | crispr_tracr | N/A (no GPU) | Not Tested | Requires nested conda env (Python 3.8 + sklearn 0.22 + 20 bioinformatics tools); arm64 needs CONDA_SUBDIR=osx-64 |
| Gene Annotation | minced | N/A (no GPU) | Not Tested | Java tool; setup.sh downloads JAR into venv bin/ |
| Gene Annotation | mmseqs | N/A (no GPU) | Working | Works |
| Gene Annotation | pyhmmer | N/A (no GPU) | Working | Works |
| Inverse Folding | ligandmpnn | latest+cu130 | Working |  |
| Inverse Folding | proteinmpnn | jax-cuda13 | Working | Auto-detects CUDA 12 vs 13; 24/25 pass (1 failure is chai, not proteinmpnn) |
| Language Models | esm2 | 2.10.0+cu130 | Working | In-process tests fail (host env missing `transformers`) but venv tool tests pass |
| Language Models | esm3 | N/A | Not Working | All tests use in-process imports; not tested via venv yet |
| Language Models | progen2 | CPU only | Not Working | Pins `torch==2.2.2` — no aarch64 CUDA wheel exists. Added `numpy<2` for numpy2 compat |
| Language Models | evo2 | N/A | Not Working | `transformer-engine` empty meta package; requires conda deps, flash-attn — x86_64 only |
| ORF Prediction | orfipy | N/A (no GPU) | Working | Fixed: run.py now uses `Path(sys.prefix) / "bin" / "orfipy"` |
| ORF Prediction | prodigal | N/A (no GPU) | Working | Works |
| RNA Splicing | splice_transformer | latest+cu130 | Working | Needed venv refresh; CPU tests always passed |
| Sequence Alignment | run_colabfold_search | N/A (no GPU) | Working | Works |
| Sequence Alignment | mafft | N/A (no GPU) | Not Working | pip package ships x86_64 ELF binaries in libexec/; platform incompatible |
| Sequence Scoring | enformer | 2.10.0+cu130 | Working | Needed venv refresh |
| Sequence Scoring | borzoi | 2.7.1+cu128 | Working | flash-attn skipped on aarch64; auto-falls back to standard borzoi model; 18/18 tests pass |
| Sequence Scoring | segmasker | N/A (no GPU) | Not Tested | Requires NCBI BLAST+ segmasker binary |
| Sequence Scoring | alphagenome | N/A | Not Working | JAX-based; `jax[cuda12_local]` doesn't support aarch64 |
| Structure Design | rfdiffusion3 | latest+cu130 | Working | 5/5 tests pass |
| Structure Dynamics | bioemu | latest+cu130 | Not Working | Setup works but inference script exit code 1; pre-existing issue |
| Structure Prediction | esmfold | 2.10.0+cu130 | Working | |
| Structure Prediction | boltz | 2.10.0+cu130 | Not Working | Setup works but inference stalls indefinitely |
| Structure Prediction | chai | 2.6.0+cu126 | Not Working | `chai_lab==0.6.1` pins `torch<2.7` which lacks sm_121 support; pre-compiled TorchScript ESM2 model incompatible |
| Structure Prediction | protenix | 2.7.1+cu128 | Not Working | Setup arch-generalized but JIT compilation fails: protenix hardcodes sm_70/sm_80, needs sm_121 |
| Structure Prediction | structure_metrics | N/A (no GPU) | Not Tested | Pure Python (BioPython); no external deps |
| Structure Prediction | viennarna | N/A (no GPU) | Working | 8/8 tests pass |
