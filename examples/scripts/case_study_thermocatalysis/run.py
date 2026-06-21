#!/usr/bin/env python3
"""
Thermocatalysis case study runner.

Runs synthesis + plot extraction on a set of PDFs, optionally evaluates
extracted plot data against human ground truth, and produces a ranking when
multiple VLMs are compared.

== Typical usage ==

Run all PDFs with default VLM (claude-sonnet-4.6):
    python run.py --pdf-dir /path/to/pdfs --output /path/to/results

Run only papers that have matching ground truth (recommended for eval):
    python run.py --pdf-dir /path/to/pdfs --output /path/to/results \\
        --gt data/results_catalysis_human --match-gt-only

Run multiple VLMs and compare:
    python run.py --pdf-dir /path/to/pdfs --output /path/to/results \\
        --gt data/results_catalysis_human --match-gt-only \\
        --vlms claude-sonnet-4.6 gemini-3-flash gpt-4o

Eval only (skip extraction, just compare existing results to GT):
    python run.py --output /path/to/results \\
        --gt data/results_catalysis_human --eval-only \\
        --vlms claude-sonnet-4.6 gemini-3-flash

Eval a single results dir (no VLM subdirs):
    python run.py --output /path/to/results/claude-sonnet-4.6 \\
        --gt data/results_catalysis_human --eval-only --single-dir

== Available VLMs ==
Any key from LLM_REGISTRY in src/llm_synthesis/utils/llms.py:
    claude-sonnet-4.6, gemini-3-flash, gpt-4o,
    qwen3.5-397b-a17b, deepseek-v3.2, gemini-2.5-flash, ...

== Output layout ==
    <output>/
        <vlm_name>/                    (omitted when --single-dir)
            <paper_id>/
                <material>.json        synthesis + performance
                performance_mappings.json
                linking_summary_llm.json
                linking_summary_human.json
            batch_summary.json
        manifest.json                  PDF→GT mapping used for this run
        vlm_ranking_rmse.json          (when multiple VLMs + GT provided)
        eval_results.csv               (when --csv and GT provided)
"""

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[3]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_repo_root / ".env", override=True)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
for _lg in ("pydantic", "LiteLLM", "litellm"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from llm_synthesis.config.domain_config import DomainConfig  # noqa: E402
from llm_synthesis.runners.batch_runner import BatchRunner  # noqa: E402

# eval helpers live alongside this script
sys.path.insert(0, str(Path(__file__).parent))
import eval_vlm  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_VLMS = ["claude-sonnet-4.6"]

GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_FALLBACK = "claude-sonnet-4-20250514"  # used only when plot_vlm unset


# ---------------------------------------------------------------------------
# GT ↔ PDF matching
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_manifest(pdf_dir: Path, gt_dir: Path | None) -> list[dict]:
    """Return list of {pdf, paper_id, gt_folder|None} for every main PDF."""
    candidates = [
        p
        for p in sorted(pdf_dir.iterdir())
        if p.suffix.lower() in (".pdf", ".md")
        and not any(
            tag in p.stem
            for tag in (
                "_SI",
                "-SI",
                "_si",
                "-si",
                "_Supporting",
                "_Supplementary",
                "_supp",
                "_Supp",
            )
        )
        and p.name != "results"
    ]
    # Dedup: if same stem has both .pdf and .md, keep .pdf
    seen: dict[str, Path] = {}
    for p in candidates:
        if p.stem not in seen or p.suffix.lower() == ".pdf":
            seen[p.stem] = p
    pdfs = sorted(seen.values(), key=lambda p: p.name)

    gt_folders = (
        {_norm(d.name): d.name for d in gt_dir.iterdir() if d.is_dir()}
        if gt_dir and gt_dir.exists()
        else {}
    )

    rows = []
    for p in pdfs:
        stem_norm = _norm(p.stem)
        gt_match = None
        for gt_norm, gt_name in gt_folders.items():
            if (
                gt_norm == stem_norm
                or gt_norm in stem_norm
                or stem_norm in gt_norm
            ):
                gt_match = gt_name
                break
        rows.append({"pdf": str(p), "paper_id": p.stem, "gt_folder": gt_match})
    return rows


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def run_extraction(
    manifest: list[dict],
    output_dir: Path,
    vlm_name: str,
    skip_existing: bool,
    max_papers: int | None,
    phase: str = "all",
    cache_dir: Path | None = None,
) -> None:
    """Run BatchRunner on the PDFs in manifest, writing to output_dir."""
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)

    runner = BatchRunner(
        domain_config=DomainConfig.for_catalysis(),
        gemini_model=GEMINI_MODEL,
        claude_model=CLAUDE_FALLBACK,
        linker_model=GEMINI_MODEL,
        synthesis_max_tokens=80_000,
        linker_max_tokens=32_000,
        plot_vlm=vlm_name,
    )

    if phase == "vlm":
        # No PDFs needed — BatchRunner reads from cache_dir
        runner.run(
            pdf_dir=output_dir,  # not used in vlm phase, just needs a path
            output_dir=output_dir,
            max_papers=max_papers,
            skip_existing=skip_existing,
            phase="vlm",
            cache_dir=cache_dir,
        )
        return

    # Write PDFs to a temp dir so BatchRunner can discover them
    with tempfile.TemporaryDirectory(prefix="catalysis_pdfs_") as tmpdir:
        tmp = Path(tmpdir)
        for row in manifest:
            src = Path(row["pdf"])
            dst = tmp / src.name
            dst.symlink_to(src)
            # Also symlink SI files if they exist (skip if already linked)
            for si_suffix in (
                "_SI",
                "-SI",
                "_si",
                "_Supporting",
                "_Supplementary",
                "_supp",
            ):
                for ext in (".pdf", ".md", ".txt", ".docx"):
                    si_src = src.parent / f"{src.stem}{si_suffix}{ext}"
                    si_dst = tmp / si_src.name
                    if si_src.exists() and not si_dst.exists():
                        si_dst.symlink_to(si_src)

        runner.run(
            pdf_dir=tmp,
            output_dir=output_dir,
            max_papers=max_papers,
            skip_existing=skip_existing,
            phase=phase,
            cache_dir=cache_dir,
        )


# ---------------------------------------------------------------------------
# Eval + reporting
# ---------------------------------------------------------------------------


def run_eval(
    results_dir: Path,
    gt_dir: Path,
    metric: str,
    vlm_label: str,
) -> list[dict]:
    rows = eval_vlm.evaluate(results_dir, gt_dir, metric=metric)
    print(f"\n--- {vlm_label} ---")
    eval_vlm._print_table(rows, metric)
    return rows


def print_ranking(ranking: list[dict], metric: str) -> None:
    print("\n" + "=" * 60)
    print(f"  VLM RANKING  —  mean {metric.upper()} (lower = better)")
    print("=" * 60)
    print(
        f"  {'Rank':<5} {'VLM':<32} {'Mean':>8}  {'Scored':>6}  {'Missing':>7}"
    )
    print("  " + "-" * 57)
    for i, row in enumerate(ranking, 1):
        mean_s = f"{row['mean']:.4f}" if row["mean"] is not None else "   N/A"
        print(
            f"  {i:<5} {row['vlm']:<32} {mean_s:>8}  "
            f"{row['n_scored']:>6}  {row['n_missing']:>7}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing catalysis PDFs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base output directory",
    )
    parser.add_argument(
        "--gt",
        default=str(_repo_root / "data" / "results_catalysis_human"),
        help="Ground truth dir (default: data/results_catalysis_human)",
    )
    parser.add_argument(
        "--vlms",
        nargs="+",
        default=DEFAULT_VLMS,
        metavar="VLM",
        help=f"VLM registry keys to run (default: {DEFAULT_VLMS}). "
        "Any key from LLM_REGISTRY in src/llm_synthesis/utils/llms.py.",
    )
    parser.add_argument(
        "--match-gt-only",
        action="store_true",
        help="Only process PDFs that have a matching ground truth folder",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae"],
        default="rmse",
        help="Error metric for eval (default: rmse)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers already processed in output dir",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max papers per VLM (for testing)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip extraction, only evaluate existing results vs GT",
    )
    parser.add_argument(
        "--single-dir",
        action="store_true",
        help="Treat --output as a single results dir (no <vlm_name>/ subdir)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip GT evaluation even if --gt is provided",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Write combined eval results to this CSV path",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "synthesis", "vlm"],
        default="all",
        help=(
            "Pipeline phase to run (default: all). "
            "'synthesis' = OCR+materials+synthesis+figures → cache only. "
            "'vlm' = load cache, run VLM plot extraction + linking. "
            "'all' = full pipeline end-to-end."
        ),
    )
    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Cache dir for synthesis phase output (default: --output). "
            "Required when --phase vlm to find previously cached synthesis."
        ),
    )
    args = parser.parse_args()

    output_base = Path(args.output).resolve()
    gt_dir = Path(args.gt).resolve() if args.gt else None
    cache_dir = Path(args.cache).resolve() if args.cache else None

    # --- Build manifest ---
    manifest = []
    if not args.eval_only:
        if args.phase != "vlm" and not args.pdf_dir:
            parser.error(
                "--pdf-dir is required unless --eval-only or --phase vlm"
            )
        pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else None
        if pdf_dir and not pdf_dir.exists():
            print(f"ERROR: --pdf-dir not found: {pdf_dir}")
            sys.exit(1)

        manifest = (
            build_manifest(pdf_dir, gt_dir if args.match_gt_only else None)
            if pdf_dir
            else []
        )

        if args.match_gt_only:
            matched = [r for r in manifest if r["gt_folder"]]
            unmatched = [r for r in manifest if not r["gt_folder"]]
            manifest = matched
        else:
            unmatched = []

        if not manifest and args.phase != "vlm":
            print("No PDFs to process (check --pdf-dir / --match-gt-only).")
            sys.exit(1)

        # Save manifest
        output_base.mkdir(parents=True, exist_ok=True)
        manifest_path = output_base / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nManifest: {len(manifest)} papers to process")
        for r in manifest:
            gt_tag = (
                f"GT: {r['gt_folder']}" if r["gt_folder"] else "no GT match"
            )
            print(f"  {r['paper_id']:<40}  [{gt_tag}]")

        if args.match_gt_only and unmatched:
            print(f"\nSkipped (no GT match): {len(unmatched)}")
            for r in unmatched:
                print(f"  {r['paper_id']}")

    # --- Extraction ---
    if not args.eval_only:
        for vlm_name in args.vlms:
            results_dir = (
                output_base if args.single_dir else output_base / vlm_name
            )
            logger.info(
                "\n%s\nVLM: %s  →  %s\n%s",
                "=" * 60,
                vlm_name,
                results_dir,
                "=" * 60,
            )
            run_extraction(
                manifest=manifest,
                output_dir=results_dir,
                vlm_name=vlm_name,
                skip_existing=args.skip_existing,
                max_papers=args.max,
                phase=args.phase,
                cache_dir=cache_dir,
            )

    # --- Eval ---
    if args.no_eval or not gt_dir or not gt_dir.exists():
        return

    print("\n\nEVALUATING vs. GROUND TRUTH")
    print(f"GT: {gt_dir}  |  Metric: {args.metric.upper()}")

    vlms_to_eval = args.vlms
    all_csv_rows = []
    ranking = []

    for vlm_name in vlms_to_eval:
        results_dir = output_base if args.single_dir else output_base / vlm_name
        if not results_dir.exists():
            logger.warning(
                "No results for %s at %s — skipping eval", vlm_name, results_dir
            )
            continue

        rows = run_eval(results_dir, gt_dir, args.metric, vlm_name)
        scored = [r for r in rows if r["score"] is not None]
        mean = sum(r["score"] for r in scored) / len(scored) if scored else None
        ranking.append(
            {
                "vlm": vlm_name,
                "mean": mean,
                "n_scored": len(scored),
                "n_missing": len(rows) - len(scored),
            }
        )
        for r in rows:
            all_csv_rows.append({"vlm": vlm_name, **r})

    if len(ranking) > 1:
        ranking.sort(key=lambda x: (x["mean"] is None, x["mean"] or 0))
        print_ranking(ranking, args.metric)

        ranking_path = output_base / f"vlm_ranking_{args.metric}.json"
        with open(ranking_path, "w") as f:
            json.dump(ranking, f, indent=2)
        print(f"\nRanking saved to {ranking_path}")

    if args.csv and all_csv_rows:
        import csv

        csv_path = Path(args.csv)
        fields = [
            "vlm",
            "paper_id",
            "material_gt",
            "material_llm",
            args.metric,
            "n_gt_series",
            "n_llm_series",
            "n_matched_series",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_csv_rows:
                w.writerow(
                    {
                        "vlm": r["vlm"],
                        "paper_id": r["paper_id"],
                        "material_gt": r["material_gt"],
                        "material_llm": r["material_llm"],
                        args.metric: r["score"],
                        "n_gt_series": r["n_gt_series"],
                        "n_llm_series": r["n_llm_series"],
                        "n_matched_series": r["n_matched_series"],
                    }
                )
        print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
