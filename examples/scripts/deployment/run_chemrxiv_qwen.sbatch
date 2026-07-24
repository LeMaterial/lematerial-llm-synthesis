#!/bin/bash
#SBATCH --job-name=synth-chemrxiv-qwen
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=300:00:00
#SBATCH --output=results/chemrxiv_qwen_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Extract (qwen3.5-397b-a17b, reasoning OFF) + judge (deepseek-v3.2) over
# full/chemrxiv. Writes ONLY a local JSONL (no --push to any HF repo); the
# LeMaterial config is published afterwards by publish_chemrxiv_recipes.sbatch.
# Resumable: load_done_ids() skips completed ids, so re-submit to continue.
# chemrxiv is ~1,962 papers, nearly all with usable text.
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split chemrxiv \
  --extractor-model qwen3.5-397b-a17b \
  --judge-model deepseek-v3.2 \
  --concurrency 96 \
  --out results/full_chemrxiv.jsonl

echo "end=$(date -u +%FT%TZ)"
