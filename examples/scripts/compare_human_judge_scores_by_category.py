import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def normalize_material_name(name: str) -> str:
    """
    Normalize material name for better matching by:
    - Converting to lowercase
    - Removing extra whitespace
    - Standardizing common separators
    - Removing common suffixes/prefixes
    """
    if not name:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = name.lower().strip()

    # Standardize separators (replace various dashes and slashes)
    normalized = re.sub(r"[–—−/−\\]", "-", normalized)  # noqa: RUF001

    # Remove common suffixes that don't affect matching
    suffixes_to_remove = [
        " single crystals",
        " crystals",
        " nanostructures",
        " nanoparticles",
        " nanorods",
        " nanowires",
        " nanoneedles",
        " nanocombs",
        " composite",
        " ceramics",
        " powders",
        " films",
        " layers",
        " samples",
        " materials",
        " compounds",
        " structures",
    ]

    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    # Remove common prefixes
    prefixes_to_remove = [
        "synthesis of ",
        "preparation of ",
        "fabrication of ",
        "formation of ",
    ]

    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

    # Clean up multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings using multiple methods.
    Returns a score between 0 and 1, where 1 is identical.
    """
    if not str1 or not str2:
        return 0.0

    # Normalize both strings
    norm1 = normalize_material_name(str1)
    norm2 = normalize_material_name(str2)

    # If normalized strings are identical, return 1.0
    if norm1 == norm2:
        return 1.0

    # Calculate sequence matcher similarity
    sequence_similarity = SequenceMatcher(None, norm1, norm2).ratio()

    # Calculate word overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        word_similarity = 0.0
    else:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        word_similarity = len(intersection) / len(union) if union else 0.0

    # Calculate substring similarity (for cases like "R3B" vs "Rhodamine 3B")
    substring_similarity = 0.0
    if len(norm1) > 3 and len(norm2) > 3:
        # Check if one is a substring of the other
        if norm1 in norm2 or norm2 in norm1:
            substring_similarity = 0.8
        else:
            # Check for significant substring matches
            for i in range(len(norm1) - 2):
                for j in range(i + 3, len(norm1) + 1):
                    substr = norm1[i:j]
                    if len(substr) >= 3 and substr in norm2:
                        max_len = max(len(norm1), len(norm2))
                        substring_similarity = max(
                            substring_similarity, len(substr) / max_len
                        )

    # Weighted combination of different similarity measures
    final_similarity = (
        0.4 * sequence_similarity
        + 0.4 * word_similarity
        + 0.2 * substring_similarity
    )

    return final_similarity


def find_best_matches(
    human_materials: list[str],
    llm_materials: list[str],
    similarity_threshold: float = 0.7,
) -> dict[str, str]:
    """
    Find the best matching pairs between human and LLM materials.
    Returns a dictionary mapping human material names to LLM material names.
    """
    matches = {}
    used_llm_materials = set()

    # Sort by similarity to prioritize better matches
    all_pairs = []
    for human_mat in human_materials:
        for llm_mat in llm_materials:
            similarity = calculate_string_similarity(human_mat, llm_mat)
            if similarity >= similarity_threshold:
                all_pairs.append((similarity, human_mat, llm_mat))

    # Sort by similarity (highest first)
    all_pairs.sort(reverse=True)

    # Assign matches greedily
    for similarity, human_mat, llm_mat in all_pairs:
        if human_mat not in matches and llm_mat not in used_llm_materials:
            matches[human_mat] = llm_mat
            used_llm_materials.add(llm_mat)

    return matches


def calculate_icc_absolute_agreement(scores1, scores2):
    """ICC(2,1): two-way random, absolute agreement, single measure (Shrout &
    Fleiss)."""
    # Check if we have enough data for ICC calculation
    if len(scores1) < 5:
        return np.nan

    df = pd.DataFrame(
        {
            "subject": np.arange(len(scores1)),
            "rater1": scores1,
            "rater2": scores2,
        }
    )
    long = pd.melt(df, id_vars="subject", var_name="rater", value_name="rating")

    try:
        icc_tbl = pg.intraclass_corr(
            data=long, targets="subject", raters="rater", ratings="rating"
        )
        # Absolute agreement, single measure → ICC2
        row = icc_tbl[icc_tbl["Type"] == "ICC2"]
        return float(row["ICC"].iloc[0]) if not row.empty else np.nan
    except (AssertionError, ValueError):
        return np.nan


def calculate_icc_consistency(scores1, scores2):
    """
    ICC(3,1): two-way mixed, consistency, single measure (Shrout & Fleiss).
    """
    # Check if we have enough data for ICC calculation
    if len(scores1) < 5:
        return np.nan

    df = pd.DataFrame(
        {
            "subject": np.arange(len(scores1)),
            "rater1": scores1,
            "rater2": scores2,
        }
    )
    long = pd.melt(df, id_vars="subject", var_name="rater", value_name="rating")

    try:
        icc_tbl = pg.intraclass_corr(
            data=long, targets="subject", raters="rater", ratings="rating"
        )
        row = icc_tbl[icc_tbl["Type"] == "ICC3"]
        return float(row["ICC"].iloc[0]) if not row.empty else np.nan
    except (AssertionError, ValueError):
        return np.nan


def aggregate_human_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates scores from multiple human experts into a single
    consensus DataFrame.
    """
    human_evals = df[df["evaluator_type"] == "human"]
    # Group by material and calculate the mean for all score columns
    human_consensus = human_evals.groupby("material_id").mean(numeric_only=True)
    return human_consensus


def evaluate_agreement_by_criterion_df(
    human_df: pd.DataFrame, llm_df: pd.DataFrame, score_columns: list[str]
) -> dict[
    str,
    tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
]:
    """
    Calculates Spearman correlation, Cohen's κ, and ICCs between human and LLM
    scores for each criterion in the ontology, matched by material_id.
    Also calculates mean, median, and standard deviation scores for both human
    and LLM.
    """
    # Column mismatch guard: Intersect score_columns with columns present in
    # both frames
    score_columns = [
        c
        for c in score_columns
        if c in human_df.columns and c in llm_df.columns
    ]

    # Merge on material_id once
    merged = pd.merge(
        human_df[["material_id", *score_columns]],
        llm_df[["material_id", *score_columns]],
        on="material_id",
        suffixes=("_human", "_llm"),
    )

    def categorize_score(v):
        # bins: (-inf,1], (1,2], (2,3], (3,4], (4, inf)
        if v <= 1:
            return 0
        elif v <= 2:
            return 1
        elif v <= 3:
            return 2
        elif v <= 4:
            return 3
        else:
            return 4

    results = {}
    for col in score_columns:
        valid = merged[[f"{col}_human", f"{col}_llm"]].dropna()
        if len(valid) < 2:
            results[col] = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            continue

        # Spearman (allow NaN when either side is constant)
        if (
            valid[f"{col}_human"].nunique() < 2
            or valid[f"{col}_llm"].nunique() < 2
        ):
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(valid[f"{col}_human"], valid[f"{col}_llm"])

        # Kappa (quadratic)
        human_categories = valid[f"{col}_human"].apply(categorize_score)
        llm_categories = valid[f"{col}_llm"].apply(categorize_score)
        kappa = cohen_kappa_score(
            human_categories, llm_categories, weights="quadratic"
        )

        # Summary stats
        human_mean = valid[f"{col}_human"].mean()
        human_median = valid[f"{col}_human"].median()
        human_std = valid[f"{col}_human"].std()
        llm_mean = valid[f"{col}_llm"].mean()
        llm_median = valid[f"{col}_llm"].median()
        llm_std = valid[f"{col}_llm"].std()

        results[col] = (
            rho,
            p,
            kappa,
            human_mean,
            human_median,
            human_std,
            llm_mean,
            llm_median,
            llm_std,
        )
    return results


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

        human_file = os.path.join(paper_dir, "result_human.json")
        llm_file = os.path.join(paper_dir, "result.json")

        # Only process if BOTH files exist
        if not (os.path.exists(human_file) and os.path.exists(llm_file)):
            skipped_papers.append(paper_id)
            continue

        processed_papers.append(paper_id)

        # Load both files to check for extraction failures
        try:
            with open(human_file) as f:
                human_evaluations = json.load(f)
            with open(llm_file) as f:
                llm_evaluations = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading files for {paper_id}: {e}")
            skipped_papers.append(f"{paper_id} (file read error)")
            continue

        # Process evaluations, skipping those with extraction failures
        print(
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
            human_synthesis = human_eval.get("synthesis", {})
            llm_synthesis = llm_eval.get("synthesis", {})

            human_target_type = human_synthesis.get("target_compound_type")
            human_synthesis_method = human_synthesis.get("synthesis_method")
            llm_target_type = llm_synthesis.get("target_compound_type")
            llm_synthesis_method = llm_synthesis.get("synthesis_method")

            # Always use LLM classifications, even if they disagree with human
            final_target_type = llm_target_type

            target_type_match = (
                human_target_type == llm_target_type
                or llm_target_type == "other"
            )
            synthesis_method_match = (
                human_synthesis_method == llm_synthesis_method
                or llm_synthesis_method == "other"
            )

            if not (target_type_match and synthesis_method_match):
                skipped_extractions.append(
                    f"{paper_id}_{human_material} (category mismatch: "
                    f"human_type='{human_target_type}' vs "
                    f"llm_type='{llm_target_type}', "
                    f"human_method='{human_synthesis_method}' vs "
                    f"llm_method='{llm_synthesis_method}')"
                )
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

    print(f"\nTotal materials with human evaluations: {len(human_df)}")
    print(f"Total materials with LLM evaluations: {len(llm_df)}")

    if skipped_extractions:
        print(
            f"\nSkipped {len(skipped_extractions)} materials due to category "
            f"mismatches:"
        )
        for extraction in skipped_extractions:
            print(f"  - {extraction}")

    return human_df, llm_df


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

    # Define score columns
    score_cols = [col for col in human_df.columns if "_score" in col]

    # Store results for all categories
    all_results = []

    print(f"\nAnalyzing by {category_column}...")
    print(f"Found {len(categories)} categories: {list(categories)}")

    for category in categories:
        if pd.isna(category) or category is None:
            continue

        print(f"\nProcessing category: {category}")

        # Filter data for this category
        human_category = human_df[human_df[category_column] == category]
        llm_category = llm_df[llm_df[category_column] == category]

        if len(human_category) == 0 or len(llm_category) == 0:
            print(f"  Skipping {category}: no data")
            continue

        print(f"  Materials in category: {len(human_category)}")

        # Calculate agreement statistics
        results = evaluate_agreement_by_criterion_df(
            human_category, llm_category, score_cols
        )

        # Store results
        for criterion, (
            corr,
            pval,
            kappa,
            human_mean,
            human_median,
            human_std,
            llm_mean,
            llm_median,
            llm_std,
        ) in results.items():
            criterion_name = (
                criterion.replace("_score", "").replace("_", " ").title()
            )

            all_results.append(
                {
                    "category": category,
                    "criterion": criterion_name,
                    "spearman": corr,
                    "p_value": pval,
                    "cohen_kappa": kappa,
                    "human_mean": human_mean,
                    "human_median": human_median,
                    "human_std": human_std,
                    "llm_mean": llm_mean,
                    "llm_median": llm_median,
                    "llm_std": llm_std,
                    "sample_size": len(human_category),
                }
            )

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

    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print(f"\nSummary by {category_column}:")
    print("=" * 80)

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

    print(summary)

    # Also print a more detailed view for categories with sufficient data
    print(
        f"\nDetailed Summary by {category_column} (categories with ≥2 samples):"
    )
    print("=" * 120)

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

        print(sufficient_data.to_string())
    else:
        print("No categories with sufficient sample size (≥2) found.")

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
        print(
            "\nMultiple human evaluators detected. "
            "Aggregating to consensus scores..."
        )
        data_human = aggregate_human_scores_df(data_human)
        # Reset index to make material_id a column again
        data_human = data_human.reset_index()

    # Analyze by target_compound_type
    print("\n" + "=" * 80)
    print("ANALYSIS BY TARGET COMPOUND TYPE")
    print("=" * 80)
    analyze_by_category(data_human, data_llm_judge, "target_compound_type")

    # Analyze by synthesis_method
    print("\n" + "=" * 80)
    print("ANALYSIS BY SYNTHESIS METHOD")
    print("=" * 80)
    analyze_by_category(data_human, data_llm_judge, "synthesis_method")

    print("\nAnalysis complete! Check the 'results' directory for CSV files.")
