#!/bin/bash
# Submit the chemrxiv inference job, then queue the publish job to start
# automatically once inference finishes SUCCESSFULLY (afterok). Fire-and-forget:
# results land in LeMaterial/LeMat-Synth-Papers once both complete.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

INFER_ID=$(sbatch --parsable examples/scripts/deployment/run_chemrxiv_qwen.sbatch)
echo "inference job:  $INFER_ID"

PUBLISH_ID=$(sbatch --parsable \
  --dependency=afterok:"$INFER_ID" \
  examples/scripts/deployment/publish_chemrxiv_recipes.sbatch)
echo "publish job:    $PUBLISH_ID  (starts after $INFER_ID succeeds)"

echo
echo "watch with: squeue --me"
