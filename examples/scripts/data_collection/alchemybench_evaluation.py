"""
AlchemyBench-style evaluation script with enhanced features.
This script demonstrates the full capabilities of the enhanced judge.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Optional pandas import for summary reports
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Summary CSV will not be generated.")

from llm_synthesis.metrics.judge.alchemybench_judge import (
    DspyAlchemyBenchSynthesisJudge,
)
from llm_synthesis.models.ontologies.alchemybench import (
    AlchemyBenchSynthesisEvaluation,
)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Instantiate the enhanced judge
    judge: DspyAlchemyBenchSynthesisJudge = instantiate(cfg.judge.architecture)
    log.info(f"Instantiated enhanced judge: {type(judge).__name__}")

    # Test cases with different material types and synthesis methods
    test_cases = get_test_cases()

    results = []

    for i, test_case in enumerate(test_cases, 1):
        log.info(f"\n=== EVALUATING TEST CASE {i}: {test_case['name']} ===")

        try:
            # Perform evaluation
            evaluation_input = (
                test_case["target_material"],
                test_case["extracted_recipe"],
                test_case["reference_procedure"],
                test_case["synthesis_context"],
            )

            result: AlchemyBenchSynthesisEvaluation = judge.forward(
                evaluation_input
            )

            # Store results
            result_data = {
                "test_case": test_case["name"],
                "target_material": test_case["target_material"],
                "overall_score": result.scores.overall_score,
                "confidence_level": result.confidence_level,
                "materials_score": (
                    result.scores.materials_appropriateness_score
                ),
                "equipment_score": (
                    result.scores.equipment_appropriateness_score
                ),
                "procedure_completeness_score": (
                    result.scores.procedure_completeness_score
                ),
                "procedure_similarity_score": (
                    result.scores.procedure_similarity_score
                ),
                "procedure_feasibility_score": (
                    result.scores.procedure_feasibility_score
                ),
                "characterization_appropriateness_score": (
                    result.scores.characterization_appropriateness_score
                ),
                "characterization_similarity_score": (
                    result.scores.characterization_similarity_score
                ),
                "critical_issues_count": len(result.critical_issues),
                "recommendations_count": len(result.recommendations),
            }
            results.append(result_data)

            # Print detailed results
            print_evaluation_results(result, test_case["name"])

            # Save individual result
            save_individual_result(result, test_case["name"], i)

        except Exception as e:
            log.error(f"Error evaluating test case {i}: {e}")
            continue

    # Generate summary report
    generate_summary_report(results)

    log.info("\n=== EVALUATION COMPLETE ===")
    log.info(f"Results saved to: {os.getcwd()}")


def get_test_cases() -> list[dict[str, str]]:
    """Define test cases covering different material types and synthesis
    methods."""

    return [
        {
            "name": "Lithium_Iron_Phosphate_Battery_Material",
            "target_material": (
                "LiFePO4/C composite for lithium-ion battery cathodes with "
                "high capacity and cycle stability"
            ),
            "synthesis_context": (
                "Material class: Phosphate composite; Application domain: "
                "Energy storage; Synthesis method: Hydrothermal"
            ),
            "reference_procedure": """
            Hydrothermal synthesis of LiFePO4/C composite:
            1. Dissolve 2.66 g LiOH·H2O, 3.58 g FeSO4·7H2O, and 2.30 g 
               NH4H2PO4 in 80 mL distilled water
            2. Add 1.0 g glucose as carbon source and stir for 30 minutes
            3. Transfer to 100 mL Teflon-lined autoclave
            4. Heat at 180°C for 12 hours
            5. Cool naturally to room temperature
            6. Filter, wash with distilled water and ethanol
            7. Dry at 80°C for 12 hours
            8. Calcine at 600°C for 2 hours under N2 atmosphere
            Characterization: XRD, SEM, TEM, electrochemical testing
            """,
            "extracted_recipe": """
            Materials: LiOH·H2O (2.5 g), FeSO4·7H2O (3.5 g), NH4H2PO4 (2.2 g), 
                       glucose (0.8 g), distilled water (75 mL)
            Equipment: Teflon-lined autoclave (100 mL), magnetic stirrer, tube 
                       furnace
            Procedure:
            1. Mix LiOH·H2O, FeSO4·7H2O, and NH4H2PO4 in water with stirring
            2. Add glucose and stir for 20 minutes
            3. Transfer to autoclave and heat at 175°C for 10 hours
            4. Cool and filter the product
            5. Wash with water and dry at 80°C
            6. Calcine at 650°C for 1 hour in nitrogen
            Characterization: XRD, SEM, electrochemical testing
            """,
        },
        {
            "name": "Titanium_Dioxide_Photocatalyst",
            "target_material": (
                "TiO2 nanoparticles with enhanced photocatalytic activity "
                "for water treatment applications"
            ),
            "synthesis_context": (
                "Material class: Metal oxide; Application domain: "
                "Environmental; Synthesis method: Sol-gel"
            ),
            "reference_procedure": """
            Sol-gel synthesis of TiO2 nanoparticles:
            1. Add 10 mL titanium tetraisopropoxide dropwise to 50 mL ethanol 
               under vigorous stirring
            2. Mix 5 mL acetic acid with 10 mL distilled water
            3. Add acid-water mixture slowly to titanium solution
            4. Stir for 2 hours at room temperature to form clear sol
            5. Age sol for 24 hours at room temperature
            6. Dry at 100°C for 12 hours to form gel
            7. Calcine at 450°C for 3 hours with 2°C/min heating rate
            Characterization: XRD, BET, UV-Vis, photocatalytic testing
            """,
            "extracted_recipe": """
            Materials: Titanium tetraisopropoxide (12 mL), ethanol (45 mL), 
                       acetic acid (4 mL), distilled water (8 mL)
            Equipment: Glass reactor, magnetic stirrer, oven, tube furnace
            Procedure:
            1. Add titanium tetraisopropoxide to ethanol with stirring
            2. Prepare acid-water solution separately
            3. Combine solutions and stir for 1 hour
            4. Age for 18 hours at room temperature
            5. Dry at 120°C overnight
            6. Calcine at 500°C for 2 hours
            Characterization: XRD, SEM, UV-Vis spectroscopy
            """,
        },
        {
            "name": "Gold_Nanoparticles_Biomedical",
            "target_material": (
                "Gold nanoparticles (10-20 nm) functionalized for drug "
                "delivery and bioimaging applications"
            ),
            "synthesis_context": (
                "Material class: Metallic nanoparticles; Application domain: "
                "Biomedical; Synthesis method: Chemical reduction"
            ),
            "reference_procedure": """
            Turkevich method for gold nanoparticle synthesis:
            1. Prepare 250 mL of 1 mM HAuCl4 solution in distilled water
            2. Heat to boiling under reflux with vigorous stirring
            3. Quickly add 25 mL of 38.8 mM sodium citrate solution
            4. Continue boiling for 15 minutes until color changes to deep red
            5. Cool to room temperature while stirring
            6. Centrifuge at 10,000 rpm for 10 minutes
            7. Wash twice with distilled water
            8. Redisperse in phosphate buffer for functionalization
            Characterization: UV-Vis, DLS, TEM, zeta potential
            """,
            "extracted_recipe": """
            Materials: HAuCl4·3H2O (0.1 g), sodium citrate (0.3 g), 
                       distilled water (300 mL)
            Equipment: Round bottom flask, reflux condenser, heating mantle, 
                       centrifuge
            Procedure:
            1. Dissolve HAuCl4 in water and heat to boiling
            2. Add sodium citrate solution rapidly
            3. Boil for 20 minutes with stirring
            4. Cool and centrifuge to collect nanoparticles
            5. Wash with distilled water
            6. Store in buffer solution
            Characterization: UV-Vis, TEM, particle size analysis
            """,
        },
    ]


def print_evaluation_results(
    result: AlchemyBenchSynthesisEvaluation, test_name: str
):
    """Print detailed evaluation results in a formatted manner."""

    print(f"\n--- DETAILED RESULTS FOR {test_name} ---")
    print(f"Overall Score: {result.scores.overall_score}/5.0")
    print(f"Confidence Level: {result.confidence_level}")

    print("\n[Individual Scores]:")
    scores = result.scores
    print(
        f"  Materials Appropriateness: "
        f"{scores.materials_appropriateness_score}/5.0"
    )
    print(
        f"  Equipment Appropriateness: "
        f"{scores.equipment_appropriateness_score}/5.0"
    )
    print(
        f"  Procedure Completeness: {scores.procedure_completeness_score}/5.0"
    )
    print(f"  Procedure Similarity: {scores.procedure_similarity_score}/5.0")
    print(f"  Procedure Feasibility: {scores.procedure_feasibility_score}/5.0")
    print(
        f"  Characterization Appropriateness: "
        f"{scores.characterization_appropriateness_score}/5.0"
    )
    print(
        f"  Characterization Similarity: "
        f"{scores.characterization_similarity_score}/5.0"
    )

    if result.critical_issues:
        print(f"\n[Critical Issues] ({len(result.critical_issues)}):")
        for issue in result.critical_issues:
            print(f"  ⚠️  {issue}")

    if result.recommendations:
        print(f"\n[Recommendations] ({len(result.recommendations)}):")
        for rec in result.recommendations:
            print(f"  💡 {rec}")

    print("\n[High-level Reasoning]:")
    print(f"{result.reasoning}")


def save_individual_result(
    result: AlchemyBenchSynthesisEvaluation, test_name: str, case_num: int
):
    """Save individual evaluation result to JSON file."""

    # Get current working directory (Hydra changes this automatically)
    current_dir = Path.cwd()
    filename = (
        current_dir
        / f"evaluation_result_{case_num:02d}_{test_name.lower()}.json"
    )

    with open(filename, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    print(f"Detailed results saved to: {filename}")


def generate_summary_report(results: list[dict[str, Any]]):
    """Generate and save summary report of all evaluations."""

    if not results:
        print("No results to summarize.")
        return

    current_dir = Path.cwd()

    print("\n=== SUMMARY REPORT ===")
    print(f"Total test cases evaluated: {len(results)}")

    # Calculate basic statistics without pandas
    overall_scores = [r["overall_score"] for r in results]
    avg_score = sum(overall_scores) / len(overall_scores)

    print(f"Average overall score: {avg_score:.2f}")

    # Count confidence levels
    confidence_counts = {}
    for r in results:
        conf = r["confidence_level"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    print("\nScore distribution by confidence level:")
    for conf, count in confidence_counts.items():
        conf_scores = [
            r["overall_score"] for r in results if r["confidence_level"] == conf
        ]
        conf_avg = sum(conf_scores) / len(conf_scores) if conf_scores else 0
        print(f"  {conf}: {conf_avg:.2f} (n={count})")

    # Calculate criterion averages
    print("\nAverage scores by criterion:")
    criterion_cols = [
        col
        for col in results[0].keys()
        if col.endswith("_score") and col != "overall_score"
    ]
    for col in criterion_cols:
        criterion_name = col.replace("_score", "").replace("_", " ").title()
        criterion_scores = [r[col] for r in results]
        criterion_avg = sum(criterion_scores) / len(criterion_scores)
        print(f"  {criterion_name}: {criterion_avg:.2f}")

    # Cases with critical issues
    print("\nTest cases with critical issues:")
    critical_cases = [r for r in results if r["critical_issues_count"] > 0]
    if critical_cases:
        for case in critical_cases:
            print(
                f"  {case['test_case']}: Score {case['overall_score']:.1f}, "
                f"Issues: {case['critical_issues_count']}"
            )
    else:
        print("  None")

    # Save detailed statistics as JSON
    stats_filename = current_dir / "alchemybench_evaluation_statistics.json"
    stats = {
        "total_cases": len(results),
        "average_overall_score": float(avg_score),
        "confidence_distribution": confidence_counts,
        "criterion_averages": {
            col.replace("_score", ""): float(
                sum(r[col] for r in results) / len(results)
            )
            for col in criterion_cols
        },
        "cases_with_critical_issues": len(critical_cases),
        "detailed_results": results,
    }

    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDetailed statistics saved to: {stats_filename}")

    # Generate CSV if pandas is available
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        summary_filename = current_dir / "alchemybench_evaluation_summary.csv"
        df.to_csv(summary_filename, index=False)
        print(f"Summary data saved to: {summary_filename}")
    else:
        # Generate simple CSV manually
        summary_filename = current_dir / "alchemybench_evaluation_summary.csv"
        with open(summary_filename, "w") as f:
            # Write header
            if results:
                header = ",".join(results[0].keys())
                f.write(header + "\n")

                # Write data rows
                for result in results:
                    row = ",".join(str(result[key]) for key in result.keys())
                    f.write(row + "\n")

        print(f"Summary CSV saved to: {summary_filename}")

    print(f"\nAll results saved in: {current_dir}")


if __name__ == "__main__":
    main()
