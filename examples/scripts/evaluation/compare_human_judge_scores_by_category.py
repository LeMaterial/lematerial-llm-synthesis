"""Compare human vs LLM judge scores -- breakdown by material category."""

import json
import logging
import os
import sys

import numpy as np
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



def read_score_data_with_categories(
    annotations_dir: str, skip_folders: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads evaluation data from the annotations directory containing
    human and LLM judgments, including category information.
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

            # Get category information

            llm_synthesis = llm_eval.get("synthesis", {})

            llm_target_type = llm_synthesis.get("target_compound_type")
            llm_synthesis_method = llm_synthesis.get("synthesis_method")

            # Always use LLM classifications, even if they disagree with human
            final_target_type = llm_target_type
            final_synthesis_method = llm_synthesis_method

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
                    "target_compound_type": final_target_type,
                    "synthesis_method": final_synthesis_method,
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
                    "target_compound_type": final_target_type,
                    "synthesis_method": final_synthesis_method,
                    "evaluator_id": "llm_judge",
                    "evaluator_type": "llm",
                }

                # Add all score fields
                for score_key, score_value in scores.items():
                    if score_key.endswith("_score"):
                        row[score_key] = score_value

                llm_data.append(row)

    # Convert to DataFrames
    human_df = pd.DataFrame(human_data)
    llm_df = pd.DataFrame(llm_data)

    logging.info(f"\nTotal materials with human evaluations: {len(human_df)}")
    logging.info(f"Total materials with LLM evaluations: {len(llm_df)}")

    if skipped_extractions:
        logging.info(
            f"\nSkipped {len(skipped_extractions)} materials due to category "
            f"mismatches:"
        )
        for extraction in skipped_extractions:
            logging.info(f"  - {extraction}")

    return human_df, llm_df


def create_score_comparison_csv(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Create a CSV with material type, synthesis type, and all score comparisons
    between LLM and human evaluations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Merge human and LLM data on material_id (include category columns)
    merge_cols = SCORE_COLUMNS + ["target_compound_type", "synthesis_method"]
    merged = merge_on_material_id(
        human_df, llm_df, merge_cols, suffixes=("_human", "_llm"),
    )

    score_columns = SCORE_COLUMNS

    # Create the comparison DataFrame
    comparison_data = []

    for _, row in merged.iterrows():
        comparison_row = {
            "material_type": row.get("target_compound_type_llm", ""),
            "synthesis_type": row.get("synthesis_method_llm", ""),
        }

        # Add all score comparisons
        for score_col in score_columns:
            human_score = row.get(f"{score_col}_human", np.nan)
            llm_score = row.get(f"{score_col}_llm", np.nan)

            comparison_row[f"{score_col}_llm"] = llm_score
            comparison_row[f"{score_col}_human"] = human_score

        comparison_data.append(comparison_row)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Save to CSV
    output_file = os.path.join(output_dir, "score_comparisons_by_material.csv")
    comparison_df.to_csv(output_file, index=False)

    logging.info(f"\nScore comparison CSV saved to: {output_file}")
    logging.info(f"Total materials in comparison: {len(comparison_df)}")

    return comparison_df


def analyze_by_category(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    category_column: str,
    output_dir: str = "results",
):
    """
    Analyze evaluation agreement by category (target_compound_type or
    synthesis_method).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get unique categories
    categories = human_df[category_column].unique()

    score_cols = SCORE_COLUMNS

    # Store results for all categories
    all_results = []

    logging.info(f"\nAnalyzing by {category_column}...")
    logging.info(f"Found {len(categories)} categories: {list(categories)}")

    for category in categories:
        if pd.isna(category) or category is None:
            continue

        logging.info(f"\nProcessing category: {category}")

        # Filter data for this category
        human_category = human_df[human_df[category_column] == category]
        llm_category = llm_df[llm_df[category_column] == category]

        if len(human_category) == 0 or len(llm_category) == 0:
            logging.info(f"  Skipping {category}: no data")
            continue

        logging.info(f"  Materials in category: {len(human_category)}")

        # Calculate agreement statistics
        results = evaluate_agreement_by_criterion(
            human_category, llm_category, score_cols, use_permutation=True,
        )

        # Store results
        for criterion, metrics in results.items():
            if metrics is None:
                continue
            all_results.append({
                "category": category,
                "criterion": col_label(criterion),
                "spearman": metrics["rho"],
                "p_value": metrics["p"],
                "cohen_kappa": metrics["kappa"],
                "human_mean": metrics["h_mean"],
                "human_median": metrics["h_median"],
                "human_std": metrics["h_std"],
                "llm_mean": metrics["l_mean"],
                "llm_median": metrics["l_median"],
                "llm_std": metrics["l_std"],
                "sample_size": metrics["n"],
            })

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)

    # Sort by sample_size (high to low) and then by category for better
    # readability
    results_df = results_df.sort_values(
        ["sample_size", "category"], ascending=[False, True]
    )

    # Save to CSV
    output_file = os.path.join(
        output_dir, f"evaluation_stats_by_{category_column}.csv"
    )
    results_df.to_csv(output_file, index=False)

    logging.info(f"\nResults saved to: {output_file}")

    # Print summary statistics
    logging.info(f"\nSummary by {category_column}:")
    logging.info("=" * 80)

    # Group by category and calculate average statistics
    summary = (
        results_df.groupby("category")
        .agg(
            {
                "spearman": "mean",
                "p_value": "mean",
                "cohen_kappa": "mean",
                "human_mean": "mean",
                "human_median": "mean",
                "human_std": "mean",
                "llm_mean": "mean",
                "llm_median": "mean",
                "llm_std": "mean",
                "sample_size": "first",
            }
        )
        .round(3)
    )

    # Sort by sample_size (high to low)
    summary = summary.sort_values("sample_size", ascending=False)

    # Reorder columns for better readability
    column_order = [
        "spearman",
        "p_value",
        "cohen_kappa",
        "human_mean",
        "human_median",
        "human_std",
        "llm_mean",
        "llm_median",
        "llm_std",
        "sample_size",
    ]
    summary = summary[column_order]

    logging.info(summary)

    # Also print a more detailed view for categories with sufficient data
    logging.info(
        f"\nDetailed Summary by {category_column} (categories with ≥2 samples):"
    )
    logging.info("=" * 120)

    # Filter for categories with sufficient sample size
    sufficient_data = summary[summary["sample_size"] >= 2].copy()

    if len(sufficient_data) > 0:
        # Format the output for better readability
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        # Round all numeric columns to 3 decimal places
        numeric_cols = sufficient_data.select_dtypes(
            include=[np.number]
        ).columns
        sufficient_data[numeric_cols] = sufficient_data[numeric_cols].round(3)

        logging.info(sufficient_data.to_string())
    else:
        logging.info("No categories with sufficient sample size (≥2) found.")

    return results_df


if __name__ == "__main__":
    # Optional: List of folders to skip entirely
    skip_folders = [
        # ### Remove deliberately bad ones
        "f2f0828a5de4a3262edc73876809a9fe03ed6ff5",
        "2883daff26f16a13134a26ca5d366549a14fcc9c",
        "90233593a9aa72b4bacfdeadc20050ae6d4b88e1",
    ]

    # Load human and LLM evaluation data with categories
    data_human, data_llm_judge = read_score_data_with_categories(
        "annotations/", skip_folders=skip_folders
    )

    # Check if we have multiple human evaluators per material
    # If so, aggregate to consensus scores
    human_counts = data_human.groupby("material_id").size()
    if (human_counts > 1).any():
        logging.info(
            "\nMultiple human evaluators detected. "
            "Aggregating to consensus scores..."
        )
        data_human = aggregate_human_scores_df(data_human)
        # Reset index to make material_id a column again
        data_human = data_human.reset_index()

    # Analyze by target_compound_type
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS BY TARGET COMPOUND TYPE")
    logging.info("=" * 80)
    analyze_by_category(data_human, data_llm_judge, "target_compound_type")

    # Analyze by synthesis_method
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS BY SYNTHESIS METHOD")
    logging.info("=" * 80)
    analyze_by_category(data_human, data_llm_judge, "synthesis_method")

    # Create score comparison CSV
    logging.info("\n" + "=" * 80)
    logging.info("CREATING SCORE COMPARISON CSV")
    logging.info("=" * 80)
    create_score_comparison_csv(data_human, data_llm_judge)

    logging.info(
        "\nAnalysis complete! Check the 'results' directory for CSV files."
    )
