#!/usr/bin/env bash
# Re-run the full superconductor corpus through the fixed pipeline
# (digitize -> filter -> link -> read, with one-shot fallback for
# materials the structured path misses -- see run_from_hf.py).
#
# The two existing large runs (results_superconductors_hf_100,
# results_superconductors_hf_random) used the OLD one-shot-only pipeline
# and need reprocessing so panels c/d (family/synthesis-method/year plots)
# reflect the same fix validated on the 19-paper ground-truth set
# (n=23, R^2=0.97, MAE=2.2 K vs. the old pipeline's honest n=27, R^2=0.54).
#
# Cost/time note: the fixed pipeline adds a digitization call (Claude) and
# a linking call (DeepSeek) per relevant plot, on top of the original
# Qwen Tc-read call, plus a one-shot fallback call for materials the
# structured path misses -- roughly 2-3x the LLM calls per figure versus
# the old pipeline. On ~1384 papers this is a genuinely large job; SHARDS
# below splits it across N parallel OS processes (each internally using
# --workers threads), and --skip-existing / --paper-timeout make it safe
# to kill and resume.
#
# Usage:
#   ./rerun_full_corpus.sh                  # all 1384 papers, 4 shards
#   ./rerun_full_corpus.sh --shards 8        # more parallel processes
#   ./rerun_full_corpus.sh --max 100         # smoke-test on 100 papers first
#
# Resuming after a kill/crash: just re-run the same command -- each shard
# passes --skip-existing, so already-completed papers (a per-paper output
# dir exists) are not reprocessed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PDF_DIR="../../../data/pdf_papers_superconductors"
OUTPUT_DIR="../../../data/results_superconductors_hf_full_v2"
SHARDS=4
WORKERS_PER_SHARD=4
PAPER_TIMEOUT=600
MAX_PAPERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards) SHARDS="$2"; shift 2 ;;
    --workers) WORKERS_PER_SHARD="$2"; shift 2 ;;
    --paper-timeout) PAPER_TIMEOUT="$2"; shift 2 ;;
    --max) MAX_PAPERS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/_logs"
mkdir -p "$LOG_DIR"

echo "=== Re-running full superconductor corpus through fixed pipeline ==="
echo "PDF dir:     $PDF_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Shards:      $SHARDS (each with --workers $WORKERS_PER_SHARD)"
echo "Timeout:     ${PAPER_TIMEOUT}s per paper"
[[ -n "$MAX_PAPERS" ]] && echo "Max papers:  $MAX_PAPERS (smoke test)"
echo

PIDS=()
for ((i = 0; i < SHARDS; i++)); do
  LOG_FILE="$LOG_DIR/shard_${i}.log"
  EXTRA_ARGS=(--skip-existing --workers "$WORKERS_PER_SHARD" \
              --paper-timeout "$PAPER_TIMEOUT" --shard "${i}/${SHARDS}")
  [[ -n "$MAX_PAPERS" ]] && EXTRA_ARGS+=(--max "$MAX_PAPERS")

  echo "Launching shard $i/$SHARDS -> $LOG_FILE"
  python3 run_from_hf.py "$PDF_DIR" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo
echo "All $SHARDS shards launched (PIDs: ${PIDS[*]}). Waiting for completion..."
echo "Tail progress with: tail -f $LOG_DIR/shard_*.log"
echo "Check failures with: grep -h 'FAILED\|TIMEOUT' $LOG_DIR/shard_*.log"

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

echo
if [[ "$FAILED" -eq 1 ]]; then
  echo "One or more shards exited with an error -- check $LOG_DIR/shard_*.log"
  exit 1
fi

echo "All shards completed. Master CSV: $OUTPUT_DIR/tc_master.csv"
echo
echo "Failed/timed-out papers (re-run this script again to retry them, "
echo "--skip-existing means only these will be reprocessed):"
grep -h "FAILED\|TIMEOUT" "$LOG_DIR"/shard_*.log || echo "  (none)"
