import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import permutation_test, spearmanr
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
    normalized = re.sub(r'[–—−/−\\]', '-', normalized)
    
    # Remove common suffixes that don't affect matching
    suffixes_to_remove = [
        ' single crystals',
        ' crystals',
        ' nanostructures',
        ' nanoparticles',
        ' nanorods',
        ' nanowires',
        ' nanoneedles',
        ' nanocombs',
        ' composite',
        ' ceramics',
        ' powders',
        ' films',
        ' layers',
        ' samples',
        ' materials',
        ' compounds',
        ' structures'
    ]
    
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    
    # Remove common prefixes
    prefixes_to_remove = [
        'synthesis of ',
        'preparation of ',
        'fabrication of ',
        'formation of '
    ]
    
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    
    # Clean up multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
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
        0.4 * sequence_similarity +
        0.4 * word_similarity +
        0.2 * substring_similarity
    )
    
    return final_similarity


def find_best_matches(human_materials: list[str], llm_materials: list[str], 
                     similarity_threshold: float = 0.7) -> dict[str, str]:
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
    df = pd.DataFrame(
        {
            "subject": np.arange(len(scores1)),
            "rater1": scores1,
            "rater2": scores2,
        }
    )
    long = pd.melt(df, id_vars="subject", var_name="rater", value_name="rating")
    icc_tbl = pg.intraclass_corr(
        data=long, targets="subject", raters="rater", ratings="rating"
    )
    # Absolute agreement, single measure → ICC2
    row = icc_tbl[icc_tbl["Type"] == "ICC2"]
    return float(row["ICC"].iloc[0]) if not row.empty else np.nan


def calculate_icc_consistency(scores1, scores2):
    """
    ICC(3,1): two-way mixed, consistency, single measure (Shrout & Fleiss).
    """
    df = pd.DataFrame(
        {
            "subject": np.arange(len(scores1)),
            "rater1": scores1,
            "rater2": scores2,
        }
    )
    long = pd.melt(df, id_vars="subject", var_name="rater", value_name="rating")
    icc_tbl = pg.intraclass_corr(
        data=long, targets="subject", raters="rater", ratings="rating"
    )
    row = icc_tbl[icc_tbl["Type"] == "ICC3"]
    return float(row["ICC"].iloc[0]) if not row.empty else np.nan


def aggregate_human_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates scores from multiple human experts into a single
    consensus DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing all evaluations,
                           with columns for
                           'paper_id', 'evaluator_id', 'evaluator_type'
                           ('human' or 'llm'), and all score columns.

    Returns:
        pd.DataFrame: A DataFrame with the mean human score for each criterion,
                      indexed by 'material_id'.
    """
    human_evals = df[df["evaluator_type"] == "human"]
    # Group by material and calculate the mean for all score columns
    human_consensus = human_evals.groupby("material_id").mean(numeric_only=True)
    return human_consensus


def print_individual_scores(
    human_df: pd.DataFrame, llm_df: pd.DataFrame, score_columns: list[str]
):
    """
    Prints individual score values for each material pair, comparing human and
    LLM scores.
    """
    # Merge on material_id to get matching pairs
    merged = pd.merge(
        human_df[["material_id", "paper_id", "material", *score_columns]],
        llm_df[["material_id", *score_columns]],
        on="material_id",
        suffixes=("_human", "_llm"),
    )

    print("\n" + "=" * 100)
    print("INDIVIDUAL SCORE COMPARISONS")
    print("=" * 100)

    for idx, row in merged.iterrows():
        paper_id = row["paper_id"]
        material_id = row["material_id"]
        material_name = row["material"]

        print(f"\nMaterial: {material_name} ({material_id})")
        print(f"Paper: {paper_id}")
        print("-" * 60)

        for col in score_columns:
            human_score = row[f"{col}_human"]
            llm_score = row[f"{col}_llm"]
            criterion_name = col.replace("_score", "").replace("_", " ").title()

            print(
                f"{criterion_name:<30} Human: {human_score:>5.1f} | "
                f"LLM: {llm_score:>5.1f} | "
                f"Diff: {abs(human_score - llm_score):>5.1f}"
            )

        print("-" * 60)


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
                np.nan,
                np.nan,
            )
            continue

        # Spearman (allow NaN when either side is constant)
        x = valid[f"{col}_human"].to_numpy()
        y = valid[f"{col}_llm"].to_numpy()

        # drop pairs with NaN/Inf
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        if (
            x.size < 2
            or np.unique(x).size < 2
            or np.unique(y).size < 2
        ):
            rho, p = np.nan, np.nan
        else:
            # Asymptotic Spearman (fast)
            res_asym = spearmanr(x, y, nan_policy="omit", 
                                alternative="two-sided")
            rho_asym, p_asym = float(res_asym.statistic), float(res_asym.pvalue)

            # Permutation-based p-value for smaller samples
            if x.size < 500:
                def stat(x_perm):
                    # permute only x relative to fixed y (pairings)
                    result = spearmanr(x_perm, y, nan_policy="omit", 
                                     alternative="two-sided")
                    return result.statistic

                res_perm = permutation_test(
                    (x,),
                    stat,
                    permutation_type="pairings",
                    n_resamples=10_000,
                    alternative="two-sided",
                    random_state=42,
                )
                # keep rho from spearmanr; use permutation p
                rho, p = rho_asym, float(res_perm.pvalue) 
                print(f"{col} permutation p-value: {p:.6g}, "
                      f"asymptotic p-value: {p_asym:.6g}")
            else:
                rho, p = rho_asym, p_asym
                
        # Kappa (quadratic)
        human_categories = valid[f"{col}_human"].apply(categorize_score)
        llm_categories = valid[f"{col}_llm"].apply(categorize_score)
        kappa = cohen_kappa_score(
            human_categories, llm_categories, weights="quadratic"
        )

        # ICCs
        icc_absolute = calculate_icc_absolute_agreement(
            valid[f"{col}_human"], valid[f"{col}_llm"]
        )
        icc_consistency = calculate_icc_consistency(
            valid[f"{col}_human"], valid[f"{col}_llm"]
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
            icc_absolute,
            icc_consistency,
            human_mean,
            human_median,
            human_std,
            llm_mean,
            llm_median,
            llm_std,
        )
    return results


def read_score_data_complete(
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
            print(f"  Human-only materials: {list(human_only)}")
        if llm_only:
            print(f"  LLM-only materials: {list(llm_only)}")
        
        # Report matches for debugging
        if matches:
            print(f"  Matched materials: {list(matches.items())}")

    # Convert to DataFrames
    human_df = pd.DataFrame(human_data)
    llm_df = pd.DataFrame(llm_data)

    # Print summary
    print(
        f"Processed {len(processed_papers)} papers with both human and "
        f"LLM evaluations:"
    )
    for paper in processed_papers:
        print(f"  - {paper}")

    if skipped_papers:
        print(f"\nSkipped {len(skipped_papers)} papers:")
        for paper in skipped_papers:
            print(f"  - {paper}")

    if skipped_extractions:
        print(
            f"\nSkipped {len(skipped_extractions)} recipe pairs due to "
            f"extraction failures:"
        )
        for extraction in skipped_extractions:
            print(f"  - {extraction}")

    print(f"\nTotal materials with human evaluations: {len(human_df)}")
    print(f"Total materials with LLM evaluations: {len(llm_df)}")

    return human_df, llm_df


if __name__ == "__main__":
    # Optional: List of folders to skip entirely
    # skip_folders = ["problematic_folder_1", "problematic_folder_2"]
    skip_folders = [
        # ### Remove deliberately bad ones
        "f2f0828a5de4a3262edc73876809a9fe03ed6ff5",
        "2883daff26f16a13134a26ca5d366549a14fcc9c",
        "90233593a9aa72b4bacfdeadc20050ae6d4b88e1"
    ]

    # Load human and LLM evaluation data (only complete pairs, no extraction
    # failures)
    data_human, data_llm_judge = read_score_data_complete(
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

    # Define which columns contain the scores to be evaluated
    score_cols = [col for col in data_human.columns if "_score" in col]

    # Print individual score comparisons
    print_individual_scores(data_human, data_llm_judge, score_cols)

    results = evaluate_agreement_by_criterion_df(
        data_human, data_llm_judge, score_cols
    )

    print("\nLLM-as-a-Judge Agreement Analysis\n")
    print(
        f"{'Criterion':<24} {'Spearman':>9} {'P-value':>9} {'Cohen κ':>8} "
        f"{'ICC(2,1)':>8} {'ICC(3,1)':>8} {'Human Mean':>11} "
        f"{'Human Median':>13} "
        f"{'Human Std':>10} {'LLM Mean':>10} {'LLM Median':>12} {'LLM Std':>9}"
    )
    print("-" * 139)

    for criterion, (
        corr,
        pval,
        kappa,
        icc_absolute,
        icc_consistency,
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
        print(
            f"{criterion_name:<24} {corr:>9.4f} {pval:>9.4f} {kappa:>8.4f} "
            f"{icc_absolute:>8.4f} {icc_consistency:>8.4f} {human_mean:>11.2f} "
            f"{human_median:>13.2f} {human_std:>10.2f} {llm_mean:>10.2f} "
            f"{llm_median:>12.2f} {llm_std:>9.2f}"
        )
