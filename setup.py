from setuptools import setup


# ===============================
# Colors
# ===============================
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"
NC = "\033[0m"
BOLD = "\033[1m"

BANNER = f"""
{BOLD}{BLUE}╔════════════════════════════════════════════════════════╗{NC}
{BOLD}{BLUE}║       Bio-Programming Tools Installation               ║{NC}
{BOLD}{BLUE}╚════════════════════════════════════════════════════════╝{NC}

{GREEN}\
    -. .-.   .-. .-.   .-. .-.   .-.
    ||\\|||\\  /|||\\|||\\  /|||\\|||\\  /|
    |/ \\|||\\|||/ \\|||\\|||/ \\|||\\|||/
    ~   `-~ `-`   `-~ `-`   `-~ `-\
{NC}

{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}

{BOLD}{GREEN}✓ bio-programming-tools installed successfully!{NC}

{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}
{BOLD}{YELLOW}  ⚠  HuggingFace Login Required for Some Models{NC}
{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}

  Some models (e.g. ESM3, Boltz, Protenix) require
  authentication with HuggingFace to download weights.

  To log in, run:
    {BOLD}huggingface-cli login{NC}

  Or set the environment variable:
    {BOLD}export HF_TOKEN=<your-token>{NC}

  Get your token at: https://huggingface.co/settings/tokens

{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}
"""

setup()
print(BANNER)
