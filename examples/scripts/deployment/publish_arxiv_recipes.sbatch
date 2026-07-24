#!/bin/bash
#SBATCH --job-name=arxiv-recipes-config
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/arxiv_recipes_config_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Join results/full_arxiv.jsonl onto source full/arxiv and publish the LeMaterial
# config "arxiv_recipes_and_evals" (same schema as omg24_recipes_and_evals: all
# paper columns + structured_synthesis[str] + evaluations[str] + 7 scores +
# overall_score; NULL where no recipe).
# Run AFTER run_arxiv_qwen.sbatch reports "Nothing to do".
# Needs an HF token with WRITE on LeMaterial. Drop --push to dry-build first.
# 62,952 rows incl. images -> the mem=96G / 12h headroom is intentional.
uv run --with datasets python \
  examples/scripts/deployment/publish_recipes_config.py --split arxiv --push

echo "end=$(date -u +%FT%TZ)"
