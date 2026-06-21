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

PDF_DIR="$REPO_ROOT/data/papers_catalysis"      # input PDFs
GT_DIR="$REPO_ROOT/data/results_catalysis_human"    # human ground truth (READ-ONLY)
CACHE_DIR="$REPO_ROOT/data/results_catalysis_cache" # phase 1 cache
RESULTS_DIR="$REPO_ROOT/data/results_catalysis"     # final VLM outputs
RANKING_CSV="$RESULTS_DIR/vlm_ranking.csv"

# VLMs to benchmark — any key from LLM_REGISTRY in src/llm_synthesis/utils/llms.py
# Comment out models whose API keys you don't have.
VLMS=(
    "gemini-3-flash"       # needs GEMINI_API_KEY
    "claude-sonnet-4.6"    # needs ANTHROPIC_API_KEY
    "qwen3.5-397b-a17b"    # needs OPENROUTER_QWEN_API_KEY
    "deepseek-v3.2"        # needs OPENROUTER_DEEPSEEK_API_KEY
    # "kimi-k2.5"          # needs OPENROUTER_KIMI_API_KEY
    # "gpt-4o"             # needs OPENAI_API_KEY
    # "gemini-2.5-flash"
    # "mistral-medium"
)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "PREFLIGHT CHECKS"
echo "============================================================"

# Check .env exists and has at least one key
ENV_FILE="$REPO_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Create it with your API keys (see README for required keys)."
    exit 1
fi

# Check PDF dir exists and has PDFs
if [[ ! -d "$PDF_DIR" ]]; then
    echo "ERROR: PDF directory not found: $PDF_DIR"
    echo "       Put your catalysis PDFs there or edit PDF_DIR at the top of this script."
    exit 1
fi
N_PDFS=$(find "$PDF_DIR" -maxdepth 1 -name "*.pdf" | wc -l | tr -d ' ')
if [[ "$N_PDFS" -eq 0 ]]; then
    echo "ERROR: No PDFs found in $PDF_DIR"
    exit 1
fi
echo "  PDFs found:        $N_PDFS  ($PDF_DIR)"

# Check ground truth exists
if [[ ! -d "$GT_DIR" ]]; then
    echo "ERROR: Ground truth dir not found: $GT_DIR"
    exit 1
fi
N_GT=$(find "$GT_DIR" -name "*_human.json" | wc -l | tr -d ' ')
echo "  GT annotations:    $N_GT  ($GT_DIR)"

# Check uv run + package importable
if ! uv run python -c "from llm_synthesis.runners.batch_runner import BatchRunner" 2>/dev/null; then
    echo "ERROR: llm_synthesis not importable."
    echo "       Run:  pip install -e '.[dev]'  from $REPO_ROOT"
    exit 1
fi
echo "  llm_synthesis:     OK"

# Check API keys for each active VLM
MISSING_KEYS=()
for VLM in "${VLMS[@]}"; do
    case "$VLM" in
        claude*)
            [[ -z "${ANTHROPIC_API_KEY:-}" ]] && grep -q "ANTHROPIC_API_KEY" "$ENV_FILE" || true
            if ! grep -q "^ANTHROPIC_API_KEY=." "$ENV_FILE" && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
                MISSING_KEYS+=("ANTHROPIC_API_KEY (needed for $VLM)")
            fi ;;
        gemini*)
            if ! grep -q "^GEMINI_API_KEY=." "$ENV_FILE" && [[ -z "${GEMINI_API_KEY:-}" ]]; then
                MISSING_KEYS+=("GEMINI_API_KEY (needed for $VLM)")
            fi ;;
        gpt*)
            if ! grep -q "^OPENAI_API_KEY=." "$ENV_FILE" && [[ -z "${OPENAI_API_KEY:-}" ]]; then
                MISSING_KEYS+=("OPENAI_API_KEY (needed for $VLM)")
            fi ;;
        qwen*)
            if ! grep -q "^OPENROUTER_QWEN_API_KEY=." "$ENV_FILE" && [[ -z "${OPENROUTER_QWEN_API_KEY:-}" ]]; then
                MISSING_KEYS+=("OPENROUTER_QWEN_API_KEY (needed for $VLM)")
            fi ;;
        deepseek*)
            if ! grep -q "^OPENROUTER_DEEPSEEK_API_KEY=." "$ENV_FILE" && [[ -z "${OPENROUTER_DEEPSEEK_API_KEY:-}" ]]; then
                MISSING_KEYS+=("OPENROUTER_DEEPSEEK_API_KEY (needed for $VLM)")
            fi ;;
        mistral*)
            if ! grep -q "^MISTRAL_API_KEY=." "$ENV_FILE" && [[ -z "${MISTRAL_API_KEY:-}" ]]; then
                MISSING_KEYS+=("MISTRAL_API_KEY (needed for $VLM)")
            fi ;;
    esac
done
# Synthesis extraction always uses Gemini; OCR always uses Mistral
if ! grep -q "^GEMINI_API_KEY=." "$ENV_FILE" && [[ -z "${GEMINI_API_KEY:-}" ]]; then
    MISSING_KEYS+=("GEMINI_API_KEY (needed for synthesis extraction)")
fi
# OCR always uses Mistral
if ! grep -q "^MISTRAL_API_KEY=." "$ENV_FILE" && [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    MISSING_KEYS+=("MISTRAL_API_KEY (needed for OCR / PDF extraction)")
fi

if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: Missing API keys in $ENV_FILE:"
    for KEY in "${MISSING_KEYS[@]}"; do
        echo "         $KEY"
    done
    echo ""
    echo "  Add them to $ENV_FILE, then re-run."
    exit 1
fi
echo "  API keys:          OK"
echo "  VLMs to run:       ${VLMS[*]}"
echo ""
echo "All checks passed. Starting pipeline..."

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

uv run run.py \
    --pdf-dir       "$PDF_DIR" \
    --output        "$CACHE_DIR" \
    --gt            "$GT_DIR" \
    --match-gt-only \
    --phase         synthesis \
    --no-eval \
    --skip-existing

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

VLM_PIDS=()
for VLM in "${VLMS[@]}"; do
    VLM_OUT="$RESULTS_DIR/$VLM"
    echo ""
    echo "--- Launching VLM (background): $VLM ---"
    echo "    Output: $VLM_OUT"

    uv run run.py \
        --output    "$VLM_OUT" \
        --phase     vlm \
        --cache     "$CACHE_DIR" \
        --vlms      "$VLM" \
        --no-eval \
        --single-dir \
        > "$VLM_OUT.log" 2>&1 &
    VLM_PIDS+=($!)
done

echo ""
echo "Waiting for all VLMs to finish (pids: ${VLM_PIDS[*]})..."
FAILED=0
for PID in "${VLM_PIDS[@]}"; do
    wait "$PID" || { echo "  WARNING: VLM process $PID exited with error"; FAILED=1; }
done
[[ "$FAILED" -eq 1 ]] && echo "  Check per-VLM logs at $RESULTS_DIR/<vlm>.log"
echo "All VLMs done."

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

uv run run.py \
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
# VISUALIZE: Generate publication figures for each VLM
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "VISUALIZE: Generating figures"
echo "============================================================"

for VLM in "${VLMS[@]}"; do
    VLM_OUT="$RESULTS_DIR/$VLM"
    FIG_OUT="$VLM_OUT/figures"
    if [[ ! -d "$VLM_OUT" ]]; then
        echo "  Skipping $VLM (no results dir)"
        continue
    fi
    echo ""
    echo "--- Figures for $VLM → $FIG_OUT ---"
    uv run catalysis_map.py "$VLM_OUT" --out-dir "$FIG_OUT" || \
        echo "  WARNING: catalysis_map.py failed for $VLM (missing matplotlib/pandas?)"
done

# ---------------------------------------------------------------------------
# DONE — print final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
echo ""
echo "Results:   $RESULTS_DIR"
echo "Ranking:   $RESULTS_DIR/vlm_ranking_rmse.json"
echo "CSV:       $RANKING_CSV"
echo "Figures:   $RESULTS_DIR/<vlm_name>/figures/"
echo ""
echo "--- VLM Ranking ---"
uv run python -c "
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
    print('  (ranking file not found)')
"

echo ""
echo "--- Re-run eval only (no extraction) ---"
echo "  uv run run.py \\"
echo "      --output $RESULTS_DIR \\"
echo "      --gt     $GT_DIR \\"
echo "      --vlms   ${VLMS[*]} \\"
echo "      --eval-only --metric rmse --csv $RANKING_CSV"

echo ""
echo "Done."
