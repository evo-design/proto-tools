# Troubleshooting

This note covers the common failure modes when running proto-tools on shared HPC
clusters and their recommended remedies, all of which apply to any cluster.
proto-tools behaves identically on a laptop and on a cluster, but shared HPC
environments impose constraints that account for most setup problems, namely an
outdated host operating system, restrictive storage quotas, batch GPU
scheduling, and locked-down networking. The underlying mechanisms are documented
in full in [tool-environments.md](tool-environments.md) and
[storage.md](storage.md).

## Standalone environment build failures

The most common cluster-specific failure is a tool's packaged `standalone/`
environment failing to install on the host machine. The usual causes are a CUDA
wheel that does not match the cluster's drivers, a pinned dependency version that
is unavailable for the platform, or an installation step that requires patching
for the host operating system.

Resolving this does not require forking proto-tools. The procedure is to diagnose
the failure and then override the environment definition.

To diagnose, enable verbose build output. Setting `PROTO_ENV_VERBOSE=1` streams
the output of `setup.sh` to stderr as the environment builds, and setting
`PROTO_ENV_LOG_DIR=<path>` preserves the complete build log even when the
environment directory is rolled back. Both are described in
[Debugging Env Setup](tool-environments.md#debugging-env-setup-proto_env_verbose-proto_env_log_dir).

To override the definition, run `proto-tools eject-standalone <toolkit>`, which
produces an editable copy of the environment definition (`setup.sh`,
`requirements.txt`, `python_version.txt`, and related files). After patching the
copy for the cluster, point proto-tools at it by exporting
`PROTO_<TOOLKIT>_STANDALONE_DIR=<path>`. The next call to that tool rebuilds the
environment from the patched copy under an isolated environment name, so the
packaged environment is never overwritten. This works whether proto-tools was
installed in editable mode or as a regular wheel. Full details are given in
[Overriding a tool's standalone env](tool-environments.md#overriding-a-tools-standalone-env-proto_toolkit_standalone_dir).

This is the self-healing path for environments: eject, patch, point, and rerun,
with no changes to the installed package.

## Incompatible host operating system

When pip-installed wheels such as numpy, torch, or jax fail to install or import
with errors such as `GLIBC_2.28 not found`, the host operating system is too old.
Older cluster login and compute nodes, for example CentOS 7 with glibc 2.17,
predate the manylinux baseline that modern machine-learning wheels target.

The remedy is to run proto-tools inside a container that provides a modern
userland while reusing the host kernel, GPU drivers, and filesystems. On HPC
systems, [Apptainer](https://apptainer.org/) (formerly Singularity) is the
conventional choice. Build the image once from a base image whose CUDA version
matches the cluster (for example `pytorch/pytorch:*-cuda*-runtime`), store the
read-only `.sif` on shared storage so that the entire group can reuse it, and
perform all installation and execution inside the container. The scheduler
continues to expose GPUs to the container through the `--nv` flag.

## Disk quota exhaustion

A `Disk quota exceeded` error usually has one of two causes. The first is tool
environments or model weights being written to a small `$HOME`. To resolve it,
redirect them off `$HOME` by setting `PROTO_HOME` (which holds tool environments
and micromamba) and `PROTO_MODEL_CACHE` (which holds model weights) to large,
persistent storage. On a cluster, `PROTO_HOME` should reside on persistent group
storage, because environments read thousands of small files at import and scratch
filesystems are periodically purged, while `PROTO_MODEL_CACHE` should reside on
fast, large storage shared with the lab so that each model is downloaded once
rather than once per user. The per-filesystem trade-offs are covered in
[storage.md](storage.md).

The second cause is the pip wheel cache filling `$HOME`. To resolve it, relocate
the cache onto scratch storage:

```bash
mkdir -p $SCRATCH/.cache && ln -sfn $SCRATCH/.cache ~/.cache
```

## Slow model loading

Slow model loading usually indicates that the weights reside on a filesystem that
performs poorly for the many small random reads that loading entails. Some
parallel filesystems are optimized for large sequential I/O instead. Move
`PROTO_MODEL_CACHE` to faster storage and delete the previous copy, and the
affected tools will re-download their weights on the next run. The per-filesystem
trade-offs are covered in [storage.md](storage.md).

## GPUs not visible to tools

Most tools require a GPU. On a scheduled cluster (SLURM or PBS), request a GPU
before running, either interactively (for example `srun ... --gpus 1 --pty bash`)
or through a batch script. When running inside a container, expose the GPU with
the `--nv` flag. The proto-tools device manager then schedules tool work across
whatever GPUs are visible.

## Gated model weights

Some tools fail to download their weights because the model is gated. To use such
a model, accept its license on the corresponding HuggingFace page and then
authenticate before running the tool:

```bash
hf auth login
# or:
export HF_TOKEN=hf_...
```

The set of gated models and the access flow for each are documented in
[Gated model access](../README.md#step-3-gated-model-access-optional) in the
README.
