import json
import os

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


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
        for idx, (human_eval, llm_eval) in enumerate(
            zip(human_evaluations, llm_evaluations)
        ):
            # Skip if either evaluation is None
            if human_eval is None or llm_eval is None:
                skipped_extractions.append(
                    f"{paper_id}_{idx} (None evaluation)"
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
                skipped_extractions.append(f"{paper_id}_{idx}")
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
                    "material_id": f"{paper_id}_{idx}",
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
                    "material_id": f"{paper_id}_{idx}",
                    "material": llm_eval.get("material", ""),
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
        "1d6ca39fff40accf64733808c749d26c30a0e4f9",
        "1df04f9e3f942b30d5e1c2bd1ab9cc3a79c23f13",
        "22fb9453271c06cc332106a3e0fda74364267b86",
        "2e268ad55e4e356b5fdf88506f15139a236282e2",
        "4fae971d628aef67a3401e06522cf59bad7fcd44",
        "5d473d45140751cd7f55c0ac5cc74284c1940d57",
        "65d95ab344eeb3fd7cf074352e8ac8a9aa57bd5c",
        "73c6aeebd5877d2eb17d4961577d98216d503e6f",
        "8c37fd10addf6d79f84ec2d5f4a8e5c6d6ef676f",
        "914dfcfe8762e189e9d7873090587458e7c86695",
        "c47e0cbc8b6feb8d28c3d9c1c29f98772ede6c27",
        ### Remove deliberately bad ones
        "1602.02498",
        "f2f0828a5de4a3262edc73876809a9fe03ed6ff5",
        "2212.12506",
        "673b3fdd7be152b1d07c21f1",
        ### LLM messed up
        "60c74548469df43eacf434a6",
        "cond-mat.0602418",
        "2404.08872",
        "2306.14755",
        "9a889c1a671fd3cae48285eaa95069d189d02fe3",
        "1902.03049",
        "0d5ffdaf23a655e1eff80bc8b6b4978067de4d5b",
        "1409.1070",
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
