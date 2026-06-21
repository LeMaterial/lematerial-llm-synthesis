#!/usr/bin/env bash
# =============================================================================
# Thermocatalysis Case Study — full walkthrough
#
# WHAT THIS DOES:
#   1. Phase 1 (once, ~30 min): OCR + material extraction + synthesis extraction
#      + figure detection → saves cache. VLM-independent, never needs to re-run.
#   2. Phase 2 (per VLM, ~5-10 min): load cache → VLM plot extraction → linking
#      → per-material JSON output. Run once per VLM.
#   3. Eval: compare each VLM's extracted plot data to human ground truth using
#      RMSE/MAE. Prints ranked table and saves JSON + CSV.
#
# PREREQUISITES:
#   - uv or pip: pip install -e ".[dev]"  (from repo root)
#   - .env at repo root with API keys:
#       ANTHROPIC_API_KEY=...
#       GEMINI_API_KEY=...
#       OPENAI_API_KEY=...
#       MISTRAL_API_KEY=...
#       OPENROUTER_QWEN_API_KEY=...
#       OPENROUTER_DEEPSEEK_API_KEY=...
#
# RUN FROM: anywhere (script sets all paths relative to repo root)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit these if your data lives elsewhere
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/examples/scripts/case_study_thermocatalysis"

PDF_DIR="$REPO_ROOT/data/ammonia_cracking_pdf"      # input PDFs
GT_DIR="$REPO_ROOT/data/results_catalysis_human"    # human ground truth (READ-ONLY)
CACHE_DIR="$REPO_ROOT/data/results_catalysis_cache" # phase 1 cache
RESULTS_DIR="$REPO_ROOT/data/results_catalysis"     # final VLM outputs
RANKING_CSV="$RESULTS_DIR/vlm_ranking.csv"

# VLMs to benchmark — any key from LLM_REGISTRY in src/llm_synthesis/utils/llms.py
# Comment out models whose API keys you don't have.
VLMS=(
    "claude-sonnet-4.6"
    "gemini-3-flash"
    "gpt-4o"
    # "qwen3.5-397b-a17b"    # needs OPENROUTER_QWEN_API_KEY
    # "deepseek-v3.2"        # needs OPENROUTER_DEEPSEEK_API_KEY
    # "gemini-2.5-flash"
    # "mistral-medium"
)

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# PHASE 1: Synthesis extraction (run ONCE, skip if cache already exists)
# ---------------------------------------------------------------------------
# Only processes the 2 papers that have human ground truth matches.
# Saves to $CACHE_DIR/_cache/<paper_id>/synthesis.json + figures.json
# Cost: ~30 min total (Gemini for synthesis, Mistral OCR).
# Re-run only if you add new papers or want to re-extract synthesis.

echo ""
echo "============================================================"
echo "PHASE 1: Synthesis extraction (OCR + materials + synthesis + figures)"
echo "Output: $CACHE_DIR/_cache/"
echo "============================================================"

python run.py \
    --pdf-dir   "$PDF_DIR" \
    --output    "$CACHE_DIR" \
    --gt        "$GT_DIR" \
    --match-gt-only \
    --phase     synthesis \
    --no-eval \
    --skip-existing   # skip papers already cached

# What was saved:
#   $CACHE_DIR/_cache/Teng_2024_Ru/synthesis.json   ← materials + synthesis + paper text
#   $CACHE_DIR/_cache/Teng_2024_Ru/figures.json     ← detected figures (base64 images)
#   $CACHE_DIR/_cache/Zhou_2021_.../synthesis.json
#   $CACHE_DIR/_cache/Zhou_2021_.../figures.json
#   $CACHE_DIR/manifest.json                        ← which PDFs ran + GT mapping

echo ""
echo "Phase 1 done. Cached papers:"
ls "$CACHE_DIR/_cache/" 2>/dev/null || echo "  (none yet)"

# ---------------------------------------------------------------------------
# PHASE 2: VLM plot extraction (run ONCE PER VLM)
# ---------------------------------------------------------------------------
# Reads cached figures, sends each to the VLM, extracts digitized plot data,
# links series to materials, writes per-material JSON.
# Cost: ~5-10 min per VLM (only VLM API calls, no OCR/synthesis).

echo ""
echo "============================================================"
echo "PHASE 2: VLM plot extraction (one run per VLM)"
echo "Output: $RESULTS_DIR/<vlm_name>/<paper_id>/<material>.json"
echo "============================================================"

for VLM in "${VLMS[@]}"; do
    VLM_OUT="$RESULTS_DIR/$VLM"
    echo ""
    echo "--- Running VLM: $VLM ---"
    echo "    Output: $VLM_OUT"

    python run.py \
        --output    "$VLM_OUT" \
        --phase     vlm \
        --cache     "$CACHE_DIR" \
        --vlms      "$VLM" \
        --no-eval \
        --single-dir

    # What was saved per paper:
    #   $VLM_OUT/<paper_id>/<material>.json            ← synthesis + performance data
    #   $VLM_OUT/<paper_id>/performance_mappings.json  ← plot→material linking
    #   $VLM_OUT/<paper_id>/linking_summary_llm.json   ← linking stats
    #   $VLM_OUT/<paper_id>/batch_summary.json         ← run summary
done

# ---------------------------------------------------------------------------
# EVAL: Compare all VLMs to human ground truth
# ---------------------------------------------------------------------------
# For each (paper, material) pair: compute RMSE between LLM-extracted
# coordinates and human-annotated coordinates.
# Matching: paper_id must match folder name; material names normalized.
# Score: RMSE=0 perfect, RMSE<0.1 good, RMSE>0.3 poor.

echo ""
echo "============================================================"
echo "EVAL: RMSE vs. human ground truth"
echo "GT: $GT_DIR"
echo "Metric: RMSE (lower = better)"
echo "============================================================"

# Build --vlms arg from array
VLM_ARGS=()
for VLM in "${VLMS[@]}"; do
    VLM_ARGS+=("$VLM")
done

python run.py \
    --output    "$RESULTS_DIR" \
    --gt        "$GT_DIR" \
    --vlms      "${VLM_ARGS[@]}" \
    --eval-only \
    --metric    rmse \
    --csv       "$RANKING_CSV"

# Output:
#   $RESULTS_DIR/vlm_ranking_rmse.json   ← VLM ranking by mean RMSE
#   $RANKING_CSV                         ← per-material scores for all VLMs

# ---------------------------------------------------------------------------
# READ RESULTS
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"

echo ""
echo "--- VLM Ranking (vlm_ranking_rmse.json) ---"
python -c "
import json, sys
path = '$RESULTS_DIR/vlm_ranking_rmse.json'
try:
    data = json.load(open(path))
    print(f'  {\"Rank\":<5} {\"VLM\":<32} {\"Mean RMSE\":>10}  {\"Scored\":>6}  {\"Missing\":>7}')
    print('  ' + '-' * 60)
    for i, row in enumerate(data, 1):
        mean = f\"{row[\"mean\"]:.4f}\" if row[\"mean\"] is not None else \"   N/A\"
        print(f'  {i:<5} {row[\"vlm\"]:<32} {mean:>10}  {row[\"n_scored\"]:>6}  {row[\"n_missing\"]:>7}')
except FileNotFoundError:
    print('  (ranking file not found — eval may not have run yet)')
"

echo ""
echo "--- Per-material scores (vlm_ranking.csv) ---"
echo "  Path: $RANKING_CSV"
echo "  Columns: vlm, paper_id, material_gt, material_llm, rmse,"
echo "           n_gt_series, n_llm_series, n_matched_series"

echo ""
echo "--- Output file structure ---"
echo "  $RESULTS_DIR/"
echo "    <vlm_name>/"
echo "      <paper_id>/"
echo "        <material>.json            ← synthesis procedure + plot_data coordinates"
echo "        performance_mappings.json  ← which plot series mapped to which material"
echo "        linking_summary_llm.json   ← linking stats (n linked, unmatched series)"
echo "        batch_summary.json         ← run timing + material counts"
echo "    vlm_ranking_rmse.json          ← final VLM ranking"
echo "    vlm_ranking.csv                ← per-material RMSE table"

echo ""
echo "--- How to read a single material result ---"
echo "  Each <material>.json has this structure:"
echo "  {"
echo "    \"material\": \"Ru/MgO(110)\","
echo "    \"synthesis\": { ...synthesis procedure... },"
echo "    \"performance\": {"
echo "      \"material_name\": \"Ru/MgO(110)\","
echo "      \"plot_data\": [{"
echo "        \"series_name\": \"Ru/MgO(110)\","
echo "        \"coordinates\": [[T, conversion], ...],"
echo "        \"x_axis_label\": \"Temperature\","
echo "        \"x_axis_unit\": \"°C\","
echo "        \"y_axis_label\": \"NH3 conversion\","
echo "        \"y_axis_unit\": \"%\""
echo "      }]"
echo "    }"
echo "  }"

echo ""
echo "--- Re-run eval only (after extracting more VLMs) ---"
echo "  python run.py \\"
echo "      --output $RESULTS_DIR \\"
echo "      --gt     $GT_DIR \\"
echo "      --vlms   claude-sonnet-4.6 gemini-3-flash gpt-4o \\"
echo "      --eval-only --metric rmse --csv $RANKING_CSV"

echo ""
echo "Done."
