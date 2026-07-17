#!/bin/bash
#SBATCH --job-name=omg24-recipes-config
#SBATCH --partition=regularcpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/omg24_recipes_config_%j.log
#SBATCH --open-mode=append
# #SBATCH --account=ACCOUNT      # uncomment + set if your cluster requires it

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

# Join the clean v2 recipes/evals/scores onto the source omg24 paper rows and
# publish them as the new LeMaterial/LeMat-Synth-Papers config
# "omg24_recipes_and_evals" (full superconductor-style schema: paper columns +
# structured_synthesis[str] + evaluations[str] + 7 scores + overall_score).
# Needs an HF token with READ on sid-betalol/*-v2 and WRITE on LeMaterial.
# Downloads the multi-GB omg24 split once (cached). Drop --push to dry-build.
uv run --with datasets python \
  examples/scripts/deployment/publish_omg24_recipes_config.py --push

echo "end=$(date -u +%FT%TZ)"
