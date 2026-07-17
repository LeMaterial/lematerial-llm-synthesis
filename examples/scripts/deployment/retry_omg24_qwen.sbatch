#!/bin/bash
#SBATCH --job-name=synth-omg24-qwen-retry
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/omg24_qwen_retry_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Retry pass: same --out as the original run, so load_done_ids() sees the 5183
# successful ids and skips them; only the ~101 rows with error != null are
# re-yielded (they already passed the paper-text filter, so they pass again).
# Fresh results are appended to the same JSONL; push_to_hub() dedups by id
# (last-wins) so the re-pushed dataset has one freshest row per paper.
# REQUIRES results/full_omg24_qwen.jsonl from the original run to be present.
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split omg24 \
  --extractor-model qwen3.5-397b-a17b \
  --judge-model deepseek-v3.2 \
  --concurrency 32 \
  --out results/full_omg24_qwen.jsonl \
  --push --hub-repo sid-betalol/LeMat-Synth-recipes-qwen-omg24

echo "end=$(date -u +%FT%TZ)"
