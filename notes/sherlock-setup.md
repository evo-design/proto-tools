<img src="https://www.sherlock.stanford.edu/assets/images/logo.png" alt="Sherlock Logo" width="120" align="left" style="margin-right: 15px;">

# Sherlock HPC Setup Guide

Setup instructions for running proto-tools on Stanford's [Sherlock cluster](https://www.sherlock.stanford.edu/docs/). Sherlock runs CentOS 7 with glibc 2.17, which is too old for most modern ML packages and a container is required. This guide walks through the full setup from scratch.

---

## 1. Container Setup

Sherlock's host OS is CentOS 7, which ships with glibc 2.17 (released 2012). Nearly every modern ML library (PyTorch, JAX, TensorFlow, and their dependencies) requires glibc 2.28+ and will fail to import on the bare host. The solution is to run inside an [Apptainer](https://apptainer.org/) (formerly Singularity) container that provides a modern Linux userland while still using the host's kernel, drivers, and filesystems.

Build the container from the official PyTorch Docker image. This must be done on a compute node; login nodes don't have enough memory for the build.

The container and conda environments (Section 3) both go on `$GROUP_HOME`. Check that your group has enough free space first (~10–15 GB needed):

```bash
df -h $GROUP_HOME
```

If `$GROUP_HOME` is nearly full, free up space or ask your PI about expanding the allocation before proceeding.

```bash
# Get on a compute node
srun -p normal --cpus-per-task 4 --mem 16G -t 1:00:00 --pty bash

# Build the .sif image (~3.5 GB)
apptainer build $GROUP_HOME/$USER/pytorch_latest.sif docker://pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

This produces a read-only `.sif` file containing Ubuntu 22.04 (glibc 2.35), Python 3.11, PyTorch 2.6.0 with CUDA support, and conda. We store it on `$GROUP_HOME` because it's fast and persistent; you'll be loading this file every time you start a session.

### Set up the shell alias

Add this to your `~/.bashrc` so you can enter the container with a single command. Adjust the `.sif` path if you stored it elsewhere:

```bash
# Add to ~/.bashrc
SIF_PATH="$GROUP_HOME/$USER/pytorch_latest.sif"  # adjust if needed
alias ptshell='APPTAINERENV_PS1="(apptainer) \u@\h:\w\$ " apptainer exec --nv --bind ~/.bashrc:/bashrc_custom --bind /share/software/user/open:/share/software/user/open:ro $SIF_PATH bash --rcfile /bashrc_custom -i'
```

What the flags do:

- `--nv` passes through NVIDIA GPU drivers from the host, so CUDA works inside the container
- `--bind ~/.bashrc:/bashrc_custom` makes your bashrc available inside the container (Apptainer doesn't source it by default), and `--rcfile /bashrc_custom` tells bash to use it
- `--bind /share/software/user/open:...` makes Sherlock's shared software installations (Node.js, Claude Code, GCC) accessible inside the container, since `module load` doesn't work in Apptainer
- All Sherlock filesystems (`$HOME`, `$GROUP_HOME`, `$OAK`, `$SCRATCH`) are automatically visible inside the container; your symlinks, data, and code all work as expected

### Verify

```bash
ptshell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
exit
```

> **Note:** This verification uses the container's built-in Python (with PyTorch). After you set up the conda env (Section 3) and activate it, `python` will point to the conda env's Python, which does **not** have PyTorch installed. That's expected. Each proto-language-tool installs PyTorch into its own isolated environment automatically via `ToolInstance`.

---

## 2. Redirect Model Weights & Caches

Sherlock limits `$HOME` (`/home/users/$USER`) to **15 GB**, so model weights and caches need to live elsewhere. By default, `PROTO_HOME` is `~/.proto/` which is under `$HOME`; you must set `PROTO_HOME` to redirect data to a larger filesystem.

Sherlock has three storage tiers available to most users:

| Filesystem | Path | Speed | Quota |
|------------|------|-------|-------|
| **Group Home** (`$GROUP_HOME`) | `/home/groups/<PI>/` | Fast (NFS) | 1 TB shared |
| **Scratch** (`$SCRATCH`) | `/scratch/users/$USER/` | Fast (lustre) | Large |
| **Oak** (`$OAK`) | `/oak/stanford/groups/<PI>/` | Slow (lustre) | Large |

**Group Home and Scratch are recommended for model weights.** Both are fast filesystems suitable for the random read patterns that model loading requires. Oak is optimized for archival storage and can be extremely slow for model loading. Oak works as a fallback but is not recommended.

Scratch auto-purges files after 90 days of **no access**; files that are read regularly are never purged. This makes it a good fit for model weights during active development, since tools re-download automatically if weights are missing.

### Model Caching Options

`PROTO_HOME` contains tool environments and micromamba (per-user), while `PROTO_MODEL_CACHE` controls model weights (can be shared). Add both to your `~/.bashrc`:

```bash
# Per-user: tool envs and micromamba (must be per-user, not shared)
export PROTO_HOME=$GROUP_HOME/$USER/proto_home

# Shared with collaborators: model weights only (safe for concurrent downloads)
export PROTO_MODEL_CACHE=$GROUP_HOME/shared_model_weights
```

**Option 1: Shared weights + per-user envs (recommended).** Share model weights on Group Home so they only download once across collaborators. Tool envs stay per-user:

```bash
export PROTO_HOME=$GROUP_HOME/$USER/proto_home
export PROTO_MODEL_CACHE=$GROUP_HOME/shared_model_weights
```

Check with your team if there's already a shared weights directory.

**Option 2: Fully per-user.** Keep everything under your own directory. Simpler but each user downloads their own copy of model weights:

```bash
export PROTO_HOME=$GROUP_HOME/$USER/proto_home
```

If Group Home is tight on space, use `$SCRATCH` instead (per-user, fast, but 90-day auto-purge).

Optionally, symlink `~/.cache` off of `$HOME` to avoid filling your 15 GB quota with pip wheel caches: `ln -sfn $SCRATCH/.cache ~/.cache`

### Verify

```bash
echo $PROTO_HOME
ls -la ~/.cache 2>/dev/null
```

---

## 3. Conda Environments

Conda environments contain thousands of small files that are read frequently during import. Where you put them matters for performance.

**Use Group Home** (`$GROUP_HOME`): it's fast NFS, persistent, and has 1 TB shared across your lab. **Never** put conda envs on Oak. Oak is a parallel lustre filesystem optimized for large sequential reads, and it's roughly 2x slower than Group Home for the many small random reads conda does. Don't use `$HOME` (too small) or `$SCRATCH` (auto-deletes after 90 days).

```bash
# Good: Group Home (fast, persistent)
conda create -p /home/groups/<PI>/$USER/envs/bio-tools python=3.11

# Bad: Oak (slow for conda), Home (too small), Scratch (auto-deletes)
```

### Register an environment directory

By default, conda only knows about environments in its default `envs/` directory, so activating a `-p` environment requires the full path every time. Register your Group Home envs directory so you can activate by name:

```bash
conda config --append envs_dirs /home/groups/<PI>/$USER/envs
```

Now `conda activate proto-tools` works from anywhere instead of typing the full path.

### Install proto-tools

Package installation **must** be done inside the container. The CentOS 7 host has glibc 2.17, which is too old for pre-built wheels of numpy, scipy, pandas, and other dependencies; they'll fail trying to build from source. Inside the container (Ubuntu 22.04, glibc 2.35), pre-built wheels install instantly.

```bash
# Enter the container first
ptshell

# Activate the env and install
conda activate proto-tools
cd /path/to/proto-tools
pip install -e ".[dev]"
```

To set up Claude Code inside the container, see [section 4](#4-claude-code-in-the-container).

---

## 4. Claude Code in the Container

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is useful for running proto-tools interactively through natural language. Getting it to work inside the Apptainer container requires some extra setup because Sherlock's `module` system (Lmod) is not available inside containers.

The `ptshell` alias already bind-mounts `/share/software/user/open` (where Sherlock installs shared software like Node.js, Claude Code, and GCC) into the container. The binaries are on the filesystem; you just need to add them to your `PATH` and `LD_LIBRARY_PATH` manually since `module load` can't do it for you.

Add this container-detection block to your `~/.bashrc`. It runs only inside the container and sets up Claude Code:

```bash
if [ -n "$APPTAINER_NAME" ] || [ -n "$SINGULARITY_NAME" ]; then
    # Claude Code + Node.js (module system unavailable inside container)
    # These globs find the latest installed version of each package
    CLAUDE_BIN=$(ls -d /share/software/user/open/claude-code/*/bin 2>/dev/null | sort -V | tail -1)
    NODE_BIN=$(ls -d /share/software/user/open/nodejs/*/bin 2>/dev/null | sort -V | tail -1)
    GCC_LIB=$(ls -d /share/software/user/open/gcc/*/lib64 2>/dev/null | sort -V | tail -1)
    [ -n "$CLAUDE_BIN" ] && export PATH="${CLAUDE_BIN}:${PATH}"
    [ -n "$NODE_BIN" ] && export PATH="${NODE_BIN}:${PATH}"
    [ -n "$GCC_LIB" ] && export LD_LIBRARY_PATH="${GCC_LIB}:${LD_LIBRARY_PATH}"
fi
```

Why each piece is needed:

- **Claude Code**: the `claude` binary itself
- **Node.js**: Claude Code is a Node.js application and won't start without it
- **GCC runtime libs**: Node.js requires a modern `libstdc++` (the one in CentOS 7 is too old). Adding GCC's `lib64` to `LD_LIBRARY_PATH` provides it

Outside the container (on login nodes), load Node.js normally via the module system:

```bash
# Add to ~/.bashrc (outside the container block)
if type module &>/dev/null; then
    module load nodejs/20.20.0
fi
```

---

## 5. GPU Sessions

Most proto-tools require a GPU. On Sherlock, you need to request GPU resources through SLURM before you can use them; GPUs are not available on login nodes.

### Interactive session

```bash
# Request a GPU node (adjust partition and time as needed)
srun -p gpu --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -t 12:00:00 --pty bash

# Enter container and activate the conda env
ptshell
conda activate proto-tools
```

If your lab has a condo partition (e.g., `brianhie`), use that for dedicated GPU access with shorter queue times:

```bash
srun -p brianhie --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -N 1 -t 12:00:00 --pty bash
```

### Batch job

For longer-running or unattended work, submit a batch job. This is useful for running tool benchmarks, generating environment reports, or processing large datasets.

Note: The `ptshell` alias uses `--rcfile` to source your bashrc, but `--rcfile` only works for interactive shells. In batch scripts, you must explicitly `source ~/.bashrc` for PATH setup and activate the conda env:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

apptainer exec --nv $GROUP_HOME/$USER/pytorch_latest.sif bash -c "
    source ~/.bashrc  # required: sets up PATH, LD_LIBRARY_PATH
    conda activate proto-tools
    python my_script.py
"
```

---

## 6. Monitoring Disk Usage

Since home directory quota is tight, it's worth periodically checking what's using space, especially after setting up new tools for the first time.

```bash
# Check home directory usage
du -sh ~
du -sh ~/.cache ~/.local 2>/dev/null

# Check Group Home and Oak usage
df -h $GROUP_HOME
lfs quota -g <PI> $OAK

# Find what's eating space in home
du -h --max-depth=2 ~ | sort -rh | head -20
```

---

## Troubleshooting

### "Disk quota exceeded" during tool setup

A model is downloading weights to `$HOME` instead of following a symlink. Check which directory grew:

```bash
du -h --max-depth=1 ~ | sort -rh | head -10
```

Check your `PROTO_HOME` setting; if unset, weights go to `~/.proto/proto_model_cache/`. See [model-weights.md](model-weights.md) for all options.

### "Disk quota exceeded" during `pip install`

pip caches wheels in `~/.cache/pip` and may install to `~/.local`. If you followed the symlink setup in step 2, both are already redirected. Otherwise:

```bash
STORAGE=/oak/stanford/groups/<PI>/projects/$USER

mkdir -p $STORAGE/.cache/pip
ln -sfn $STORAGE/.cache/pip ~/.cache/pip

mkdir -p $STORAGE/.local
ln -sfn $STORAGE/.local ~/.local
```

### Model loads slowly

If model loading is very slow, your weights are likely on Oak. Move them to Group Home or Scratch (see [Section 2](#2-redirect-model-weights--caches)). Update `PROTO_HOME` (or `PROTO_MODEL_CACHE`) and delete the old weights directory; tools will re-download to the new location on next run.
