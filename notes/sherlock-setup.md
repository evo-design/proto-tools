<img src="https://www.sherlock.stanford.edu/assets/images/logo.png" alt="Sherlock Logo" width="120" align="left" style="margin-right: 15px;">

# Sherlock HPC Setup Guide

Setup instructions for running proto-tools on Stanford's [Sherlock cluster](https://www.sherlock.stanford.edu/docs/). Sherlock runs CentOS 7 with glibc 2.17, which is too old for most modern ML packages, so a container is required.

## Quick Start

Run these in order. Each step is explained in detail below; come back here once you've done it once.

```bash
# 1. Build the Apptainer image — skip if a labmate already did it
#    (the .sif is read-only and shareable across the group).
srun -p normal --cpus-per-task 4 --mem 16G -t 1:00:00 --pty bash
mkdir -p $GROUP_HOME/simg
apptainer build $GROUP_HOME/simg/pytorch_latest.sif \
    docker://pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
exit

# 2. Edit ~/.bashrc — add the block from "Step 2: Configure ~/.bashrc" below.

# 3. Reload your shell, then enter the container and install proto-tools:
source ~/.bashrc
ptshell
conda create -p $GROUP_HOME/$USER/envs/proto-tools python=3.11 -y
conda activate proto-tools
cd /path/to/proto-tools && pip install -e ".[dev]"

# 4. Verify Claude Code works:
claude --version
```

For GPU work, swap step 3's `srun` for a GPU partition (see [Step 5](#step-5-gpu-sessions)).

---

## Step 1: Build the Apptainer Container

Sherlock's host OS is CentOS 7 with glibc 2.17 (released 2012). Modern ML libraries (PyTorch, JAX, TensorFlow) need glibc 2.28+. The fix is an [Apptainer](https://apptainer.org/) container that ships a modern Linux userland while reusing the host's kernel, GPU drivers, and filesystems.

The result is a read-only `.sif` file shareable across your whole group — only one person needs to build it. Check first:

```bash
ls $GROUP_HOME/simg/pytorch_latest.sif 2>/dev/null && echo "already built — skip to Step 2"
```

If it's not there, build it. The build needs ~3.5 GB of disk and must run on a **compute node** — login nodes don't have enough memory.

```bash
# Check $GROUP_HOME has space for the .sif and conda envs (~10–15 GB total)
df -h $GROUP_HOME

# Get on a compute node and build to a group-shared location
srun -p normal --cpus-per-task 4 --mem 16G -t 1:00:00 --pty bash
mkdir -p $GROUP_HOME/simg
apptainer build $GROUP_HOME/simg/pytorch_latest.sif \
    docker://pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

The image contains Ubuntu 22.04 (glibc 2.35), Python 3.11, PyTorch 2.6.0 + CUDA, and conda. Stored on `$GROUP_HOME/simg/` so the whole lab shares one copy and it loads fast every session.

---

## Step 2: Configure `~/.bashrc`

The block below adds `ptshell` and the rest of the setup to your shell. **`ptshell` is a one-word alias for "enter the Apptainer container"**: instead of typing the long `apptainer exec --nv --bind … pytorch_latest.sif bash …` invocation every time, you just run `ptshell` and you're inside the container with GPU access, your `.bashrc` sourced, and Sherlock's shared software visible. You'll use it any time you want to run Python or pip against proto-tools.

The block also redirects proto-tools data off your 15 GB `$HOME`, and sets up Claude Code so it works both on login nodes (via Lmod) and inside the container (via direct `PATH` lookups, since `module` doesn't exist there).

```bash
# ─── proto-tools / Sherlock setup ───

# 1. Container shell alias (group-shared .sif by default)
SIF_PATH="$GROUP_HOME/simg/pytorch_latest.sif"   # adjust if your lab stores it elsewhere
alias ptshell='APPTAINERENV_PS1="(apptainer) \u@\h:\w\$ " apptainer exec --nv \
    --bind ~/.bashrc:/bashrc_custom \
    --bind /share/software/user/open:/share/software/user/open:ro \
    $SIF_PATH bash --rcfile /bashrc_custom -i'

# 2. Storage: keep proto-tools data off $HOME (15 GB quota).
#    Tool envs go on $GROUP_HOME (persistent, 1 TB, per-user).
#    Model weights go on $GROUP_SCRATCH so the whole lab shares one copy
#    (fast lustre, large, ~10–100s of GB per model). STRONGLY RECOMMENDED:
#    coordinate with your lab on a shared path so each model only downloads
#    once across collaborators instead of once per user.
export PROTO_HOME=$GROUP_HOME/$USER/proto_home
export PROTO_MODEL_CACHE=$GROUP_SCRATCH/proto_model_cache
# Fallback if you can't share (per-user scratch — same speed, no sharing):
# export PROTO_MODEL_CACHE=$SCRATCH/proto_model_cache

# 3. Claude Code on login nodes (when Lmod is healthy)
if type module &>/dev/null; then
    module load claude-code 2>/dev/null
    module load nodejs/20.20.0 2>/dev/null
fi

# 4. Claude Code fallback — works inside the container, and on login nodes
#    when Lmod is broken (it can be in some shells, e.g. Claude Code's own
#    shell snapshot). Adds the latest installed version of each tool to PATH.
if ! command -v claude &>/dev/null; then
    CLAUDE_BIN=$(ls -d /share/software/user/open/claude-code/*/bin 2>/dev/null | sort -V | tail -1)
    NODE_BIN=$(ls -d /share/software/user/open/nodejs/*/bin 2>/dev/null | sort -V | tail -1)
    GCC_LIB=$(ls -d /share/software/user/open/gcc/*/lib64 2>/dev/null | sort -V | tail -1)
    [ -n "$CLAUDE_BIN" ] && export PATH="${CLAUDE_BIN}:${PATH}"
    [ -n "$NODE_BIN" ]   && export PATH="${NODE_BIN}:${PATH}"
    [ -n "$GCC_LIB" ]    && export LD_LIBRARY_PATH="${GCC_LIB}:${LD_LIBRARY_PATH}"
fi
```

Why each piece is needed:

- **`ptshell`**: enters the container with GPU passthrough (`--nv`), your `.bashrc` (so aliases and env vars carry over), and `/share/software/user/open` (so you can find Sherlock's installed Node.js, Claude Code, and GCC inside the container — `module` doesn't work there).
- **`PROTO_HOME`** (tool envs + micromamba): redirects off your 15 GB `$HOME` to `$GROUP_HOME`, which is fast NFS, persistent, and 1 TB shared with the lab. These need to be persistent — `$SCRATCH` would purge them.
- **`PROTO_MODEL_CACHE`** (model weights): goes on `$GROUP_SCRATCH` so the lab shares one copy. Models are large (often 10–100s of GB each) and re-downloading per-user wastes time, network, and quota. Lustre is fast for the random reads model loading does. Both `$GROUP_SCRATCH` and `$SCRATCH` auto-purge after 90 days of **no access**, but actively-used weights are read often and are never purged — and if a tool's weights ever do get cleaned up, it just re-downloads them. Coordinate with your lab on the path before setting it. Avoid `$OAK` for model weights (~2x slower for small-file random reads).
- **Claude Code module loads** (block #3): work on login nodes when Lmod is functional. The `2>/dev/null` swallows errors so a broken Lmod doesn't spam the shell.
- **Claude Code fallback** (block #4): kicks in when `claude` isn't on `PATH` after the module loads — which happens inside the container (no Lmod) and in any shell where Lmod is broken. Globs find the latest installed versions and add them directly to `PATH`. Node.js + GCC libs are needed because Claude Code is a Node.js app, and Node needs a modern `libstdc++` that CentOS 7's runtime doesn't provide.

After editing, reload:

```bash
source ~/.bashrc
claude --version    # should print a version, both on login nodes and inside ptshell
```

Optional: symlink `~/.cache` off `$HOME` to keep pip wheels off your quota:

```bash
ln -sfn $SCRATCH/.cache ~/.cache
```

---

## Step 3: Conda Environment

Conda envs read thousands of small files at import time, so place them on **`$GROUP_HOME`** (fast NFS, persistent, 1 TB). Never on `$OAK` (~2x slower for small reads), `$HOME` (too small), or `$SCRATCH` (90-day auto-purge).

Package installation **must happen inside the container** — pre-built wheels (numpy, scipy, etc.) target glibc 2.28+, so they fail to install on the bare CentOS 7 host.

```bash
# Register your envs directory once so `conda activate <name>` works
conda config --append envs_dirs $GROUP_HOME/$USER/envs

# Enter the container and create the env
ptshell
conda create -p $GROUP_HOME/$USER/envs/proto-tools python=3.11 -y
conda activate proto-tools

# Install proto-tools
cd /path/to/proto-tools
pip install -e ".[dev]"
```

**Note:** the container's built-in `python` ships with PyTorch, but the conda env you just made does not — that's fine. Each proto-tool installs its own PyTorch into an isolated environment automatically via `ToolInstance`.

---

## Step 4: HuggingFace Auth (Optional)

Some tools use gated HF models. To use them, accept the license on the model page, then either:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
# or:
export HF_TOKEN=hf_...
```

| Model | HF Repo |
|-------|---------|
| ESM3 | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) |
| AlphaGenome | [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) |

---

## Step 5: GPU Sessions

Most proto-tools require a GPU. Request one through SLURM before entering the container:

```bash
# General GPU partition
srun -p gpu --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -t 12:00:00 --pty bash

# Or your PI's condo partition (faster queues, dedicated hardware)
srun -p <your-pi-partition> --gpus 1 --cpus-per-task 8 --mem-per-cpu=30GB -N 1 -t 12:00:00 --pty bash

ptshell
conda activate proto-tools
```

For batch jobs, `--rcfile` doesn't apply (it's interactive-only), so source `.bashrc` explicitly:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

apptainer exec --nv $GROUP_HOME/$USER/pytorch_latest.sif bash -c "
    source ~/.bashrc
    conda activate proto-tools
    python my_script.py
"
```

---

## Troubleshooting

### `claude: command not found`

The fallback in Step 2 should handle this, but if it's still missing:

```bash
ls /share/software/user/open/claude-code/   # confirm Claude Code is installed
command -v claude                           # check what Lmod loaded
```

If Claude Code is installed but not on `PATH`, the fallback block didn't run — confirm your `.bashrc` block is below the conda init lines and sourced (`source ~/.bashrc`).

### `module: command not found` or `bad interpreter: /bin/lua`

Lmod is broken in this shell (happens in some Claude Code subshells). Block #3 in Step 2 silently no-ops; block #4 (the fallback) kicks in. If `claude` still isn't on `PATH`, run the fallback manually:

```bash
export PATH=$(ls -d /share/software/user/open/claude-code/*/bin | sort -V | tail -1):$PATH
export PATH=$(ls -d /share/software/user/open/nodejs/*/bin | sort -V | tail -1):$PATH
```

### `Disk quota exceeded` during tool setup

A model is downloading weights to `$HOME` instead of `$PROTO_HOME`. Check:

```bash
echo $PROTO_HOME                   # should be set
du -h --max-depth=1 ~ | sort -rh | head -10
```

See [storage.md](storage.md) for full options.

### `Disk quota exceeded` during `pip install`

Pip caches wheels in `~/.cache/pip`. Symlink it off `$HOME`:

```bash
mkdir -p $SCRATCH/.cache && ln -sfn $SCRATCH/.cache ~/.cache
```

### Model loads slowly

Weights are likely on `$OAK`. Move `$PROTO_MODEL_CACHE` to `$GROUP_HOME` or `$SCRATCH` and delete the old directory; tools will re-download on next run.

### Disk usage check

```bash
du -sh ~ ~/.cache ~/.local 2>/dev/null
df -h $GROUP_HOME
lfs quota -g <PI> $OAK
```
