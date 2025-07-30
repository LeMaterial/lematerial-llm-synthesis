import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import List, Dict, Tuple
import json
import os


def aggregate_human_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates scores from multiple human experts into a single consensus DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing all evaluations, with columns for
                           'paper_id', 'evaluator_id', 'evaluator_type' ('human' or 'llm'),
                           and all score columns.

    Returns:
        pd.DataFrame: A DataFrame with the mean human score for each criterion,
                      indexed by 'paper_id'.
    """
    human_evals = df[df['evaluator_type'] == 'human']
    # Group by paper and calculate the mean for all score columns
    human_consensus = human_evals.groupby('paper_id').mean(numeric_only=True)
    return human_consensus

def evaluate_agreement_by_criterion_df(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    score_columns: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates Spearman correlation between human and LLM scores
    for each criterion thats in the ontology, matched by material_id.
    """
    # Merge on material_id once
    merged = pd.merge(
        human_df[['material_id'] + score_columns],
        llm_df[['material_id'] + score_columns],
        on='material_id',
        suffixes=('_human', '_llm')
    )

    results: Dict[str, Tuple[float, float]] = {}
    for col in score_columns:
        # Drop any pairs with missing data
        valid = merged[[f"{col}_human", f"{col}_llm"]].dropna()
        if len(valid) < 2:
            results[col] = (np.nan, np.nan)
        else:
            # If all columns are the same, set the correlation to 1, spearmanr will return nan
            if valid[f"{col}_human"].nunique() == 1 and valid[f"{col}_llm"].nunique() == 1:
                rho = 1.0
                p = 0.0
            else:
                 rho, p = spearmanr(valid[f"{col}_human"], valid[f"{col}_llm"])
                 # Fix NaN values that may occur when all values are the same
                 if np.isnan(p):
                     p = 0.0

            results[col] = (rho, p)

    return results

def read_score_data(annotations_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads evaluation data from the annotations directory containing human and LLM judgments.
    
    Args:
        annotations_dir (str): Path to the annotations directory containing paper subdirectories
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Human evaluations DataFrame and LLM evaluations DataFrame
    """
    human_data = []
    llm_data = []
    
    # Iterate through each paper directory
    for paper_id in os.listdir(annotations_dir):
        paper_dir = os.path.join(annotations_dir, paper_id)
        
        # Skip if not a directory
        if not os.path.isdir(paper_dir):
            continue
            
        human_file = os.path.join(paper_dir, 'result_human.json')
        llm_file = os.path.join(paper_dir, 'result.json')
        
        # Process human evaluations
        if os.path.exists(human_file):
            try:
                with open(human_file, 'r') as f:
                    human_evaluations = json.load(f)
                
                for idx, eval_data in enumerate(human_evaluations):
                    if 'evaluation' in eval_data and 'scores' in eval_data['evaluation']:
                        scores = eval_data['evaluation']['scores']
                        
                        # Create a row for this evaluation
                        row = {
                            'paper_id': paper_id,
                            'material_id': f"{paper_id}_{idx}",
                            'material': eval_data.get('material', ''),
                            'evaluator_id': 'human_expert',
                            'evaluator_type': 'human'
                        }
                        
                        # Add all score fields
                        for score_key, score_value in scores.items():
                            if score_key.endswith('_score'):
                                row[score_key] = score_value
                        
                        human_data.append(row)
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading human file {human_file}: {e}")
        
        # Process LLM evaluations
        if os.path.exists(llm_file):
            try:
                with open(llm_file, 'r') as f:
                    llm_evaluations = json.load(f)
                
                for idx, eval_data in enumerate(llm_evaluations):
                    if 'evaluation' in eval_data and 'scores' in eval_data['evaluation']:
                        scores = eval_data['evaluation']['scores']
                        
                        # Create a row for this evaluation
                        row = {
                            'paper_id': paper_id,
                            'material_id': f"{paper_id}_{idx}",
                            'material': eval_data.get('material', ''),
                            'evaluator_id': 'llm_judge',
                            'evaluator_type': 'llm'
                        }
                        
                        # Add all score fields
                        for score_key, score_value in scores.items():
                            if score_key.endswith('_score'):
                                row[score_key] = score_value
                        
                        llm_data.append(row)
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading LLM file {llm_file}: {e}")
    
    # Convert to DataFrames
    human_df = pd.DataFrame(human_data)
    llm_df = pd.DataFrame(llm_data)
    
    return human_df, llm_df

if __name__ == "__main__":
    # Load human and LLM evaluation data
    data_human, data_llm_judge = read_score_data('annotations/')
    
    # Define which columns contain the scores to be evaluated
    score_cols = [col for col in data_human.columns if '_score' in col]

    results = evaluate_agreement_by_criterion_df(data_human, data_llm_judge, score_cols)

    print("LLM-as-a-Judge Agreement Analysis (Spearman's Rank Correlation)\n")
    print(f"{'Criterion':<45} {'Spearman Coefficient':<25} {'P-value'}")
    print("-" * 80)

    for criterion, (corr, pval) in results.items():
        criterion_name = criterion.replace('_score', '').replace('_', ' ').title()
        print(f"{criterion_name:<45} {corr:<25.4f} {pval:.4f}")