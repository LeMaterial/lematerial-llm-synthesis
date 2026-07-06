#!/bin/bash
#SBATCH --job-name=synth-extract
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=300:00:00
#SBATCH --output=results/superconductor_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

# Make uv available in the non-login batch shell.
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# If the internet test showed compute nodes need a proxy, set it here:
# export http_proxy=http://PROXY:PORT
# export https_proxy=http://PROXY:PORT

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# I/O-bound job (API calls). --concurrency sizes the thread pool directly,
# so 8 CPUs is plenty. Resumable: rerun this script to continue where it
# stopped (already-completed paper ids are skipped).
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config superconductor_keywords_and_LLM --split full \
  --concurrency 32 \
  --push --hub-repo sid-betalol/LeMat-Synth-recipes-dummy

echo "end=$(date -u +%FT%TZ)"
