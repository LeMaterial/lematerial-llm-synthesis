#!/bin/bash
#SBATCH --job-name=chemrxiv-recipes-config
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/chemrxiv_recipes_config_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Join results/full_chemrxiv.jsonl onto source full/chemrxiv and publish the
# LeMaterial config "chemrxiv_recipes_and_evals" (same schema as
# omg24_recipes_and_evals: all paper columns + structured_synthesis[str] +
# evaluations[str] + 7 scores + overall_score; NULL where no recipe).
# Run AFTER run_chemrxiv_qwen.sbatch reports "Nothing to do".
# Needs an HF token with WRITE on LeMaterial. Drop --push to dry-build first.
uv run --with datasets python \
  examples/scripts/deployment/publish_recipes_config.py --split chemrxiv --push

echo "end=$(date -u +%FT%TZ)"
