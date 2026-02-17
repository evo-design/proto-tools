# Environment Dev Notes: Platform Compatibility
This file contains notes on platform compatibility with our current `setup.sh` scripts and `ToolInstance` system.

## Validating Venvs on a New Platform

Use `--run-all-venvs` to run one smoke test per standalone venv tool. This is the fastest way to check which tools work on a new machine.

```bash
# All venv smoke tests (CPU + GPU)
pytest --run-all-venvs

# CPU-only venv smoke tests
pytest --run-all-venvs --cpu

# List without running
pytest --run-all-venvs --collect-only
```

The flag overrides `slow`, `skip_ci`, and CPU/GPU filtering. Each tool has exactly one test marked `@pytest.mark.run_all_venvs`.

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
| ORF Prediction | prodigal | N/A (no GPU) | Working | 28/28 tests pass |
| RNA Splicing | splice_transformer | latest+cu126 | Working | 2/2 tests pass (CPU and GPU) |
| Sequence Alignment | colabfold_search (local) | N/A (no GPU) | Not Working | 17/21 tests pass (4 failures); colabfold_search exit code 1 errors |
| Sequence Alignment | colabfold_search (remote) | N/A (no GPU) | Working | 2/2 tests pass |
| Sequence Alignment | mafft | N/A (no GPU) | Working | 40/40 tests pass |
| Sequence Scoring | enformer | 2.6.0+cu126 | Working | 8/8 tests pass |
| Sequence Scoring | borzoi | 2.6.0+cu126 | Working | 14/14 tests pass |
| Sequence Scoring | segmasker | N/A (no GPU) | Not Tested | Requires NCBI BLAST+ segmasker binary |
| Sequence Scoring | alphagenome | N/A | Not Working | 6/8 tests pass (2 failures); worker errors and timeout on interval prediction |
| Structure Design | rfdiffusion3 | latest+cu126 | Not Working | 2/3 tests pass (1 failure); worker error on unconditional design |
| Structure Dynamics | bioemu | latest+cu126 | Working | 13/13 tests pass |
| Structure Prediction | esmfold | 2.6.0+cu126 | Working | Included in structure_prediction tests (30/40 passed overall) |
| Structure Prediction | boltz | 2.10.0+cu130 | Working | Included in structure_prediction tests (30/40 passed overall) |
| Structure Prediction | chai | 2.6.0+cu126 | Not Working | 0/10 tests pass (10 failures); ImportError: pyarrow missing, timeouts, invalid JSON |
| Structure Prediction | protenix | 2.7.1+cu128 | Working | 9/9 tests pass |
| Structure Prediction | alphafold3 | N/A | Not Working | 0/12 tests (12 errors); ValueError: Structure content is invalid |
| Structure Prediction | structure_metrics | N/A (no GPU) | Not Working | 10/11 tests pass (1 failure); worker error on PDB test |
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
**Last Updated:** 2026-02-15 (`pytest -sv --all`: 938 passed, 108 failed, 25 skipped in 2h08m)

| Category | Tool | GPU Version | Status | Notes |
|----------|------|-------------|--------|-------|
| Causal Models | evo1 | 2.7.1+cu128 | Not Working | Venv builds; model loading fails (StripedHyena init error). 26/30 pass (unit tests only), 4 integration fail |
| Gene Annotation | blast | N/A (no GPU) | Working | 36/36 tests pass |
| Gene Annotation | crispr_tracr | N/A (no GPU) | Not Working | Venv setup fails; requires x86_64 bioconda packages (vmatch, etc.). 22/26 pass (unit tests only), 4 integration fail |
| Gene Annotation | minced | N/A (no GPU) | Not Working | Venv builds but integration tests return empty results. 18/20 pass (unit tests only), 2 integration fail |
| Gene Annotation | mmseqs | N/A (no GPU) | Working | 29/29 tests pass |
| Gene Annotation | pyhmmer | N/A (no GPU) | Working | 21/21 tests pass |
| Inverse Folding | ligandmpnn | 2.7.1+cu128 | Working | 2/2 tests pass |
| Inverse Folding | proteinmpnn | jax 0.9.0.1 | Working | 11/11 tests pass |
| Language Models | esm2 | 2.10.0+cu130 | Working | 11/11 tests pass |
| Language Models | esm3 | 2.10.0+cu130 | Working | 12/12 tests pass |
| Language Models | progen2 | N/A | Not Working | Venv setup fails; pins `torch==2.2.2` — no aarch64 CUDA wheel exists. 5/30 pass (unit tests only), 25 integration fail |
| Language Models | evo2 | N/A | Not Working | Venv setup fails; `transformer-engine` empty meta package; requires conda deps, flash-attn — x86_64 only. 6/32 pass (unit tests only), 18 fail, 8 skip |
| ORF Prediction | orfipy | N/A (no GPU) | Working | 19/19 tests pass |
| ORF Prediction | prodigal | N/A (no GPU) | Working | 28/29 pass, 1 skip |
| RNA Splicing | splice_transformer | 2.10.0+cu130 | Working | 2/2 tests pass |
| Sequence Alignment | colabfold_search | N/A (no GPU) | Not Working | Venv builds but `mmseqs` binary not found at runtime (`FileNotFoundError: PosixPath('mmseqs')`). 17/22 pass (unit tests only), 4 integration fail, 1 skip |
| Sequence Alignment | mafft | N/A (no GPU) | Working | 40/40 tests pass; compiled from source on aarch64 (v7.525) |
| Sequence Scoring | borzoi | 2.7.1+cu128 | Working | 14/14 tests pass |
| Sequence Scoring | enformer | 2.10.0+cu130 | Working | 8/8 tests pass |
| Sequence Scoring | segmasker | N/A (no GPU) | Not Tested | Requires NCBI BLAST+ segmasker binary; no dedicated tests in suite |
| Sequence Scoring | alphagenome | N/A | Not Working | Venv builds but all tests fail. 0/8 pass |
| Structure Design | rfdiffusion3 | 2.7.1+cu128 | Not Working | Venv builds but unconditional design fails (subprocess CalledProcessError). 2/3 pass (unit tests only), 1 integration fail |
| Structure Dynamics | bioemu | N/A | Not Tested | 13/13 tests pass but all mock `ToolInstance.dispatch`; venv never built or tested on this platform |
| Structure Prediction | alphafold3 | N/A | Not Tested | 2/4 pass (unit tests), 2 skip (requires API key) |
| Structure Prediction | boltz2 | 2.10.0+cu130 | Not Working | All 10 folding tests timeout after 600s |
| Structure Prediction | chai1 | N/A | Not Working | Venv setup fails; `chai_lab` pins `torch<2.7` which lacks sm_121 support. pyarrow import error on rebuilt venv. 0/10 fail |
| Structure Prediction | esmfold | 2.10.0+cu130 | Working | 4/4 folding tests pass |
| Structure Prediction | protenix | 2.7.1+cu128 | Not Working | setup.sh returns exit 1 on rebuild. 0/21 pass (9 in test_protenix + 12 parametrized) |
| Structure Prediction | structure_metrics | N/A (no GPU) | Not Working | Venv builds but integration test fails (`ValueError: Input dict must contain an 'operation' key for legacy dispatch`). 10/11 pass (unit tests only), 1 integration fail |
| Structure Prediction | viennarna | N/A (no GPU) | Working | 6/6 tests pass |

### DGX Spark: Regressions to Investigate

The following tools regressed between 2026-02-13 and 2026-02-15 (`pytest -sv --all` on `bv/persistence`). Venvs were rebuilt from scratch during the run, so these may be caused by dispatch/worker interface changes on this branch.

- [ ] **minced** (20/20 → 18/20): Integration tests return empty results after venv rebuild. Venv setup completes successfully but `run_minced` produces no CRISPR array output.
- [ ] **colabfold_search** (21/22 → 17/22): `FileNotFoundError: PosixPath('mmseqs')` — the colabfold_search venv can't find the mmseqs binary at runtime. Likely a PATH issue in the rebuilt venv.
- [ ] **rfdiffusion3** (3/3 → 2/3): Unconditional design fails with `subprocess.CalledProcessError` from `rfd3 design`. Venv builds and unit tests pass; only the GPU integration test fails.
- [ ] **structure_metrics** (11/11 → 10/11): Integration test fails with `ValueError: Input dict must contain an 'operation' key for legacy dispatch`. Suggests the standalone inference.py needs updating for the new dispatch protocol.
