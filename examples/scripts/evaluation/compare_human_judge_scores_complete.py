"""Compare human vs LLM judge scores -- complete (all materials pooled)."""

import json
import logging
import os
import sys

import pandas as pd

from eval_utils import (
    SCORE_COLUMNS,
    aggregate_human_scores_df,
    col_label,
    evaluate_agreement_by_criterion,
    find_best_matches,
    merge_on_material_id,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_individual_scores(human_df, llm_df, score_columns):
    """Log side-by-side human vs LLM scores for every matched material."""
    merged = merge_on_material_id(
        human_df, llm_df, score_columns, suffixes=("_human", "_llm"),
    )
    logging.info("\n%s\nINDIVIDUAL SCORE COMPARISONS\n%s", "=" * 100, "=" * 100)
    for _, row in merged.iterrows():
        logging.info(
            "\nMaterial: %s (%s) | Paper: %s",
            row.get("material", ""), row["material_id"],
            row.get("paper_id", ""),
        )
        logging.info("-" * 60)
        for c in score_columns:
            h, l = row.get(f"{c}_human"), row.get(f"{c}_llm")
            if h is not None and l is not None:
                logging.info(
                    "  %-28s Human: %5.1f | LLM: %5.1f | Diff: %+5.1f",
                    col_label(c), h, l, l - h,
                )


def read_score_data(
    annotations_dir: str, skip_folders: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads evaluation data from the annotations directory containing
    human and LLM judgments. Only processes folders that have BOTH
    result.json and result_human.json files, and skips recipe pairs
    where extraction failed in either file.

    Args:
        annotations_dir (str): Path to the annotations directory containing
                                paper subdirectories
        skip_folders (list[str], optional): List of folder names to skip
                                            entirely. If None or empty, no
                                            folders are skipped.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Human evaluations DataFrame and
                                           LLM evaluations DataFrame
    """
    if skip_folders is None:
        skip_folders = []

    human_data = []
    llm_data = []
    processed_papers = []
    skipped_papers = []
    skipped_extractions = []

    # Iterate through each paper directory
    for paper_id in os.listdir(annotations_dir):
        paper_dir = os.path.join(annotations_dir, paper_id)

        # Skip if not a directory
        if not os.path.isdir(paper_dir):
            continue

        # Skip if folder is in skip_folders list
        if paper_id in skip_folders:
            skipped_papers.append(f"{paper_id} (manually skipped)")
            continue

        human_file = os.path.join(paper_dir, "old", "result_human.json")
        llm_file = os.path.join(paper_dir, "old", "result.json")

        # Only process if BOTH files exist
        if not (os.path.exists(human_file) and os.path.exists(llm_file)):
            skipped_papers.append(paper_id)
            continue

        processed_papers.append(paper_id)

        # Load both files to check for extraction failures
        try:
            with open(human_file, encoding="utf-8") as f:
                human_evaluations = json.load(f)
            with open(llm_file, encoding="utf-8") as f:
                llm_evaluations = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logging.info(f"Error reading files for {paper_id}: {e}")
            skipped_papers.append(f"{paper_id} (file read error)")
            continue

        # Process evaluations, skipping those with extraction failures
        logging.info(
            f"Processing {paper_id}: {len(human_evaluations)} human evals, "
            f"{len(llm_evaluations)} LLM evals"
        )

        # Create dictionaries to match evaluations by material name
        human_eval_dict = {}
        llm_eval_dict = {}

        # Index human evaluations by material name
        for idx, human_eval in enumerate(human_evaluations):
            if human_eval is not None:
                material_name = human_eval.get("material", f"unknown_{idx}")
                human_eval_dict[material_name] = (idx, human_eval)

        # Index LLM evaluations by material name
        for idx, llm_eval in enumerate(llm_evaluations):
            if llm_eval is not None:
                material_name = llm_eval.get("material", f"unknown_{idx}")
                llm_eval_dict[material_name] = (idx, llm_eval)

        # Use fuzzy matching to find best matches
        human_materials = list(human_eval_dict.keys())
        llm_materials = list(llm_eval_dict.keys())

        # Find best matches using fuzzy string matching
        matches = find_best_matches(
            human_materials, llm_materials, similarity_threshold=0.7
        )

        # Process matched materials
        for human_material, llm_material in matches.items():
            human_idx, human_eval = human_eval_dict[human_material]
            llm_idx, llm_eval = llm_eval_dict[llm_material]

            # Skip if either evaluation is None
            if human_eval is None or llm_eval is None:
                skipped_extractions.append(
                    f"{paper_id}_{human_material} (None evaluation)"
                )
                continue

            # Check for extraction failures in either file
            human_notes = human_eval.get("synthesis", {}).get("notes", "")
            llm_notes = llm_eval.get("synthesis", {}).get("notes", "")

            # Convert None to empty string to avoid TypeError
            human_notes = "" if human_notes is None else str(human_notes)
            llm_notes = "" if llm_notes is None else str(llm_notes)

            # Skip if extraction failed in either file
            if (
                "Extraction failed:" in human_notes
                or "Extraction failed:" in llm_notes
            ):
                skipped_extractions.append(f"{paper_id}_{human_material}")
                continue

            # Process human evaluation
            if (
                human_eval is not None
                and "evaluation" in human_eval
                and "scores" in human_eval["evaluation"]
            ):
                scores = human_eval["evaluation"]["scores"]

                # Create a row for this evaluation
                row = {
                    "paper_id": paper_id,
                    "material_id": f"{paper_id}_{human_material}",
                    "material": human_eval.get("material", ""),
                    "evaluator_id": "human_expert",
                    "evaluator_type": "human",
                }

                # Add all score fields
                for score_key, score_value in scores.items():
                    if score_key.endswith("_score"):
                        row[score_key] = score_value

                human_data.append(row)

            # Process LLM evaluation
            if (
                llm_eval is not None
                and "evaluation" in llm_eval
                and "scores" in llm_eval["evaluation"]
            ):
                scores = llm_eval["evaluation"]["scores"]

                # Create a row for this evaluation
                row = {
                    "paper_id": paper_id,
                    "material_id": f"{paper_id}_{human_material}",
                    "material": llm_eval.get("material", ""),
                    "evaluator_id": "llm_judge",
                    "evaluator_type": "llm",
                }

                # Add all score fields
                for score_key, score_value in scores.items():
                    if score_key.endswith("_score"):
                        row[score_key] = score_value

                llm_data.append(row)

        # Report unmatched materials
        matched_human_materials = set(matches.keys())
        matched_llm_materials = set(matches.values())
        human_only = set(human_eval_dict.keys()) - matched_human_materials
        llm_only = set(llm_eval_dict.keys()) - matched_llm_materials

        if human_only:
            logging.info(f"  Human-only materials: {list(human_only)}")
        if llm_only:
            logging.info(f"  LLM-only materials: {list(llm_only)}")

        # Report matches for debugging
        if matches:
            logging.info(f"  Matched materials: {list(matches.items())}")

    # Convert to DataFrames
    human_df = pd.DataFrame(human_data)
    llm_df = pd.DataFrame(llm_data)

    # Print summary
    logging.info(
        f"Processed {len(processed_papers)} papers with both human and "
        f"LLM evaluations:"
    )
    for paper in processed_papers:
        logging.info(f"  - {paper}")

    if skipped_papers:
        logging.info(f"\nSkipped {len(skipped_papers)} papers:")
        for paper in skipped_papers:
            logging.info(f"  - {paper}")

    if skipped_extractions:
        logging.info(
            f"\nSkipped {len(skipped_extractions)} recipe pairs due to "
            f"extraction failures:"
        )
        for extraction in skipped_extractions:
            logging.info(f"  - {extraction}")

    logging.info(f"\nTotal materials with human evaluations: {len(human_df)}")
    logging.info(f"Total materials with LLM evaluations: {len(llm_df)}")

    return human_df, llm_df


if __name__ == "__main__":
    logging.basicConfig(
        filename="results/human_judge_complete.log",
        level=logging.INFO, filemode="w",
    )

    skip_folders = [
        # ### Remove deliberately bad ones

        "f2f0828a5de4a3262edc73876809a9fe03ed6ff5",
        "2883daff26f16a13134a26ca5d366549a14fcc9c",
        "90233593a9aa72b4bacfdeadc20050ae6d4b88e1",
    ]

    data_human, data_llm_judge = read_score_data(
        "annotations/", skip_folders=skip_folders
    )

    # Aggregate multiple human evaluators to consensus
    human_counts = data_human.groupby("material_id").size()
    if (human_counts > 1).any():
        logging.info(
            "\nMultiple human evaluators detected. Aggregating to consensus..."
        )
        data_human = aggregate_human_scores_df(data_human)
        data_human = data_human.reset_index()

    score_cols = SCORE_COLUMNS

    print_individual_scores(data_human, data_llm_judge, score_cols)

    results = evaluate_agreement_by_criterion(
        data_human, data_llm_judge, score_cols, use_permutation=True,
    )

    logging.info("\nLLM-as-a-Judge Agreement Analysis\n")
    header = (
        f"{'Criterion':<24} {'Spearman':>9} {'P-value':>9} {'Cohen k':>8} "
        f"{'ICC(2,1)':>8} {'ICC(3,1)':>8} {'H-Mean':>7} {'H-Med':>6} "
        f"{'H-Std':>6} {'L-Mean':>7} {'L-Med':>6} {'L-Std':>6}"
    )
    logging.info("%s\n%s", header, "-" * len(header))

    for criterion, metrics in results.items():
        if metrics is None:
            continue
        logging.info(
            "%-24s %9.4f %9.4f %8.4f %8.4f %8.4f %7.2f %6.2f %6.2f %7.2f %6.2f %6.2f",
            col_label(criterion),
            metrics["rho"], metrics["p"], metrics["kappa"], metrics["icc2"], metrics["icc3"],
            metrics["h_mean"], metrics["h_median"], metrics["h_std"],
            metrics["l_mean"], metrics["l_median"], metrics["l_std"],
        )
