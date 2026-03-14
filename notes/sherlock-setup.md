<img src="https://www.sherlock.stanford.edu/assets/images/logo.png" alt="Sherlock Logo" width="120" align="left" style="margin-right: 15px;">

# Sherlock HPC Setup Guide

Setup instructions for running bio-programming-tools on Stanford's [Sherlock cluster](https://www.sherlock.stanford.edu/docs/). Sherlock runs CentOS 7 with glibc 2.17, which is too old for most modern ML packages and a container is required. This guide walks through the full setup from scratch.

---

## 1. Container Setup

Sherlock's host OS is CentOS 7, which ships with glibc 2.17 (released 2012). Nearly every modern ML library — PyTorch, JAX, TensorFlow, and their dependencies — requires glibc 2.28+ and will fail to import on the bare host. The solution is to run inside an [Apptainer](https://apptainer.org/) (formerly Singularity) container that provides a modern Linux userland while still using the host's kernel, drivers, and filesystems.

Build the container from the official PyTorch Docker image. This must be done on a compute node — login nodes don't have enough memory for the build.

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

This produces a read-only `.sif` file containing Ubuntu 22.04 (glibc 2.35), Python 3.11, PyTorch 2.6.0 with CUDA support, and conda. We store it on `$GROUP_HOME` because it's fast and persistent — you'll be loading this file every time you start a session.

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
- All Sherlock filesystems (`$HOME`, `$GROUP_HOME`, `$OAK`, `$SCRATCH`) are automatically visible inside the container — your symlinks, data, and code all work as expected

### Verify

```bash
ptshell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
exit
```

> **Note:** This verification uses the container's built-in Python (with PyTorch). After you set up the conda env (Section 3) and auto-activation (Section 4), `python` will point to the conda env's Python, which does **not** have PyTorch installed — that's expected. Each bio-programming-tool installs PyTorch into its own isolated environment automatically via `ToolInstance`.

---

## 2. Redirect Cache Directories

Sherlock limits `$HOME` (`/home/users/$USER`) to **15 GB**. Many ML tools download multi-GB model weights to subdirectories of `$HOME` on first run — a single large model (e.g., Evo2-7B at ~14 GB) can fill your entire home directory. You **must** redirect these caches to larger storage before running any tools, or you will hit "Disk quota exceeded" errors.

Sherlock provides several filesystems with different speed/size tradeoffs:

| Filesystem | Path | Speed | Quota | Use for |
|------------|------|-------|-------|---------|
| **Group Home** (`$GROUP_HOME`) | `/home/groups/<PI>/` | Fast (NFS) | 1 TB shared | Conda envs, tool venvs, container .sif |
| **Oak** (`$OAK`) | `/oak/stanford/groups/<PI>/` | Slow (lustre) | Large | Model weights, checkpoints, caches |
| **Scratch** (`$SCRATCH`) | `/scratch/users/$USER/` | Fast (lustre) | Large | Temp work (**90-day auto-purge**) |

Model weights are a good fit for Oak: they're large, read sequentially once at load time, and the slow filesystem doesn't matter after the initial read (the OS page cache keeps them in memory for subsequent loads). Conda environments need fast random reads across many small files, so they belong on Group Home.

### Set up symlinks

Follow the instructions in **[model-weights-cache.md](model-weights-cache.md)** to symlink cache directories from `$HOME` to either `$GROUP_HOME` or `$OAK`. That guide documents every directory that tools write to and provides copy-paste symlink commands.

On Sherlock, use `$OAK` as the storage target for model weights and caches. The path below may not exist yet — `mkdir -p` will create it:

```bash
STORAGE=/oak/stanford/groups/<PI>/projects/$USER
mkdir -p $STORAGE
```

Or use `$GROUP_HOME` if your lab doesn't have Oak allocation:

```bash
STORAGE=/home/groups/<PI>/$USER
mkdir -p $STORAGE
```

The simplest approach is to redirect all of `~/.cache` at once (covers HuggingFace, torch hub, pip, and tool environments):

```bash
mv ~/.cache ~/.cache.bak
mkdir -p $STORAGE/.cache
ln -sfn $STORAGE/.cache ~/.cache
cp -a ~/.cache.bak/* ~/.cache/ 2>/dev/null || true
# Verify the copy before deleting the backup
du -sh ~/.cache ~/.cache.bak
rm -rf ~/.cache.bak
```

Then redirect the remaining directories that live outside `~/.cache` — see [model-weights-cache.md](model-weights-cache.md) for the full list.

### Verify symlinks

```bash
ls -la ~/.cache ~/.local ~/.model_cache ~/.foundry 2>/dev/null
```

Each should show `->` pointing to your storage target, not be a real directory.

---

## 3. Conda Environments

Conda environments contain thousands of small files that are read frequently during import. Where you put them matters for performance.

**Use Group Home** (`$GROUP_HOME`) — it's fast NFS, persistent, and has 1 TB shared across your lab. **Never** put conda envs on Oak — Oak is a parallel lustre filesystem optimized for large sequential reads, and it's roughly 2x slower than Group Home for the many small random reads conda does. Don't use `$HOME` (too small) or `$SCRATCH` (auto-deletes after 90 days).

```bash
# Good: Group Home (fast, persistent)
conda create -p /home/groups/<PI>/$USER/envs/bio-tools python=3.11

# Bad: Oak (slow for conda), Home (too small), Scratch (auto-deletes)
```

### Install bio-programming-tools

Package installation **must** be done inside the container. The CentOS 7 host has glibc 2.17, which is too old for pre-built wheels of numpy, scipy, pandas, and other dependencies — they'll fail trying to build from source. Inside the container (Ubuntu 22.04, glibc 2.35), pre-built wheels install instantly.

```bash
# Enter the container first
ptshell

# Activate the env and install
conda activate /home/groups/<PI>/$USER/envs/bio-tools
cd /path/to/bio-programming-tools
pip install -e ".[dev]"
pre-commit install
```

To auto-activate inside the container, see the `~/.bashrc` block in [section 4 (Claude Code)](#4-claude-code-in-the-container) — it handles both conda activation and Claude Code setup.

---

## 4. Claude Code in the Container

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is useful for running bio-programming-tools interactively through natural language. Getting it to work inside the Apptainer container requires some extra setup because Sherlock's `module` system (Lmod) is not available inside containers.

The `ptshell` alias already bind-mounts `/share/software/user/open` (where Sherlock installs shared software like Node.js, Claude Code, and GCC) into the container. The binaries are on the filesystem — you just need to add them to your `PATH` and `LD_LIBRARY_PATH` manually since `module load` can't do it for you.

Add this container-detection block to your `~/.bashrc`. It runs only inside the container and handles both conda activation and Claude Code setup:

```bash
if [ -n "$APPTAINER_NAME" ] || [ -n "$SINGULARITY_NAME" ]; then
    conda activate /home/groups/<PI>/$USER/envs/bio-tools

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

- **Claude Code** — the `claude` binary itself
- **Node.js** — Claude Code is a Node.js application and won't start without it
- **GCC runtime libs** — Node.js requires a modern `libstdc++` (the one in CentOS 7 is too old). Adding GCC's `lib64` to `LD_LIBRARY_PATH` provides it

Outside the container (on login nodes), load Node.js normally via the module system:

```bash
# Add to ~/.bashrc (outside the container block)
if type module &>/dev/null; then
    module load nodejs/20.20.0
fi
```

---

## 5. GPU Sessions

Most bio-programming-tools require a GPU. On Sherlock, you need to request GPU resources through SLURM before you can use them — GPUs are not available on login nodes.

### Interactive session

```bash
# Request a GPU node (adjust partition and time as needed)
srun -p gpu --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -t 12:00:00 --pty bash

# Enter container (conda env is auto-activated by the bashrc block from Section 4)
ptshell
```

If your lab has a condo partition (e.g., `brianhie`), use that for dedicated GPU access with shorter queue times:

```bash
srun -p brianhie --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -N 1 -t 12:00:00 --pty bash
```

### Batch job

For longer-running or unattended work, submit a batch job. This is useful for running tool benchmarks, generating environment reports, or processing large datasets.

Note: The `ptshell` alias uses `--rcfile` to source your bashrc, but `--rcfile` only works for interactive shells. In batch scripts, you must explicitly `source ~/.bashrc` to get conda activation and PATH setup:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

apptainer exec --nv $GROUP_HOME/$USER/pytorch_latest.sif bash -c "
    source ~/.bashrc  # required — sets up conda env, PATH, LD_LIBRARY_PATH
    python my_script.py
"
```

---

## 6. Monitoring Disk Usage

Since home directory quota is tight, it's worth periodically checking what's using space — especially after setting up new tools for the first time.

```bash
# Check home directory usage
du -sh ~
du -sh ~/.cache ~/.local ~/.model_cache 2>/dev/null

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

Then symlink the offending directory to your storage target. See [model-weights-cache.md](model-weights-cache.md) for the full list of directories tools write to.

### "Disk quota exceeded" during `pip install`

pip caches wheels in `~/.cache/pip` and may install to `~/.local`. If you followed the symlink setup in step 2, both are already redirected. Otherwise:

```bash
STORAGE=/oak/stanford/groups/<PI>/projects/$USER

mkdir -p $STORAGE/.cache/pip
ln -sfn $STORAGE/.cache/pip ~/.cache/pip

mkdir -p $STORAGE/.local
ln -sfn $STORAGE/.local ~/.local
```

### Model loads slowly from Oak

First load from Oak is slow; subsequent loads use the OS page cache and are fast. This is expected. For frequently-loaded models where even the first load matters, consider copying weights to `$SCRATCH` (fast lustre, but remember the 90-day auto-purge):

```bash
cp -r ~/.cache/huggingface/hub/models--facebook--esmfold_v1 $SCRATCH/.cache/huggingface/hub/
```
