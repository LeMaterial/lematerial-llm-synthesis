#!/bin/bash
#SBATCH --job-name=synth-omg24-qwen
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=300:00:00
#SBATCH --output=results/omg24_qwen_%j.log
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

# Extract (qwen3.5-397b-a17b) + judge (deepseek-v3.2) over full/omg24. Both run
# through OpenRouter: the extractor uses OPENROUTER_QWEN_API_KEY, the judge uses
# OPENROUTER_DEEPSEEK_API_KEY. Pushes RESULT columns to a personal repo (copy
# over to LeMat later); the source dataset is untouched.
# Separate --out and --hub-repo from the Claude run so resume-by-id and --push
# do not mix the two model runs. Resumable: rerun to continue (ids are skipped).
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split omg24 \
  --extractor-model qwen3.5-397b-a17b \
  --judge-model deepseek-v3.2 \
  --concurrency 96 \
  --out results/full_omg24_qwen.jsonl \
  --push --hub-repo sid-betalol/LeMat-Synth-recipes-qwen-omg24

echo "end=$(date -u +%FT%TZ)"
