#!/bin/bash
#SBATCH --job-name=synth-omg24-qwen-clean
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=300:00:00
#SBATCH --output=results/omg24_qwen_clean_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Full CLEAN re-run of full/omg24 now that qwen reasoning is actually disabled
# (reasoning.effort="none" reaches the provider — the extractors no longer strip
# extra_body). The prior run had reasoning silently ON, which truncated some
# material lists (101 hard errors + ~4% silently under-extracted, incl. rows
# collapsed to n=1). This supersedes that dataset.
#
# NOTE: fresh --out (…_clean.jsonl), NOT the old file — reusing the old --out
# would let load_done_ids() skip the 5183 possibly-truncated rows and defeat the
# whole point. This processes every paper from scratch.
#
# Pushes to a SEPARATE -v2 repo so the original dataset is left untouched; both
# coexist and you can promote/replace after verifying the clean run. The push
# only runs AFTER the job succeeds. To overwrite the original instead, change
# --hub-repo back to sid-betalol/LeMat-Synth-recipes-qwen-omg24 (HF keeps commit
# history, so the old version stays recoverable even then).
uv run --with datasets python examples/scripts/deployment/run_extract_judge_hf.py \
  --config full --split omg24 \
  --extractor-model qwen3.5-397b-a17b \
  --judge-model deepseek-v3.2 \
  --concurrency 96 \
  --out results/full_omg24_qwen_clean.jsonl \
  --push --hub-repo sid-betalol/LeMat-Synth-recipes-qwen-omg24-v2

echo "end=$(date -u +%FT%TZ)"
