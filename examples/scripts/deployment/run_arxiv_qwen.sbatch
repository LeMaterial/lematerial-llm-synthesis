#!/bin/bash
#SBATCH --job-name=synth-arxiv-qwen
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=300:00:00
#SBATCH --output=results/arxiv_qwen_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Extract (qwen3.5-397b-a17b, reasoning OFF) + judge (deepseek-v3.2) over
# full/arxiv. Writes ONLY a local JSONL (no --push); the LeMaterial config is
# published afterwards by publish_arxiv_recipes.sbatch.
#
# BIG RUN: ~62,952 papers, ~638M context tokens (~3.2B effective input) —
# roughly 10x omg24. It almost certainly will not finish in one walltime window;
# it is RESUMABLE (load_done_ids() skips completed ids), so just re-submit this
# script until the log reports "Nothing to do."
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split arxiv \
  --extractor-model qwen3.5-397b-a17b \
  --judge-model deepseek-v3.2 \
  --concurrency 96 \
  --out results/full_arxiv.jsonl

echo "end=$(date -u +%FT%TZ)"
