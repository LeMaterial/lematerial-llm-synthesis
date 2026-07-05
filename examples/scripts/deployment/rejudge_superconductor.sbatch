#!/bin/bash
#SBATCH --job-name=synth-rejudge
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=300:00:00
#SBATCH --output=results/rejudge_%j.log
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

# Judge-only rerun: reuse recipes already in the dummy repo, re-run ONLY the
# deepseek judge to backfill the full evaluation (per-dimension reasonings +
# missing_information / extraction_errors / improvement_suggestions), then push
# the completed rows back. No Claude extraction happens.
# NOTE: writes to a fresh JSONL so it does not merge with the extract run's file.
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config superconductor_keywords_and_LLM --split full \
  --rejudge-from sid-betalol/LeMat-Synth-recipes-dummy \
  --concurrency 32 \
  --out results/superconductor_rejudge.jsonl \
  --push --hub-repo sid-betalol/LeMat-Synth-recipes-dummy

echo "end=$(date -u +%FT%TZ)"
