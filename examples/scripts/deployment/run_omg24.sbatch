#!/bin/bash
#SBATCH --job-name=synth-omg24
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=300:00:00
#SBATCH --output=results/omg24_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# If compute nodes need a proxy, set it here:
# export http_proxy=http://PROXY:PORT
# export https_proxy=http://PROXY:PORT

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Extract (claude-sonnet-4.6) + judge (deepseek-v3.2) over full/omg24 (16,026
# papers). Pushes RESULT columns to a dummy repo; the canonical dataset is
# augmented separately with augment_source_with_results.py after this finishes.
# Resumable: rerun to continue (completed ids are skipped).
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split omg24 \
  --concurrency 64 \
  --out results/full_omg24.jsonl \
  --push --hub-repo LeMaterial/LeMat-Synth-recipes-dummy

echo "end=$(date -u +%FT%TZ)"
