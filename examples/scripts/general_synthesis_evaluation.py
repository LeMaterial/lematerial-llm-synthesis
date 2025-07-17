"""
General Synthesis Ontology evaluation script.
This script demonstrates how to use the GeneralSynthesisOntology judge.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    GeneralSynthesisEvaluation,
)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Instantiate the judge
    judge: DspyGeneralSynthesisJudge = instantiate(cfg.judge.architecture)
    log.info(f"Instantiated judge: {type(judge).__name__}")

    # Test cases with different synthesis types
    test_cases = get_test_cases()
    results = []

    for i, test_case in enumerate(test_cases, 1):
        log.info(f"\n=== EVALUATING TEST CASE {i}: {test_case['name']} ===")

        try:
            # Perform evaluation
            evaluation_input = (
                test_case["source_text"],
                test_case["extracted_ontology_json"],
                test_case["target_material"],
            )

            result: GeneralSynthesisEvaluation = judge.forward(
                evaluation_input
            )

            # Store results
            result_data = {
                "test_case": test_case["name"],
                "target_material": test_case["target_material"],
                "overall_score": result.scores.overall_score,
                "confidence_level": result.confidence_level,
                "structural_completeness_score": (
                    result.scores.structural_completeness_score
                ),
                "material_extraction_score": (
                    result.scores.material_extraction_score
                ),
                "process_steps_score": result.scores.process_steps_score,
                "equipment_extraction_score": (
                    result.scores.equipment_extraction_score
                ),
                "conditions_extraction_score": (
                    result.scores.conditions_extraction_score
                ),
                "semantic_accuracy_score": (
                    result.scores.semantic_accuracy_score
                ),
                "format_compliance_score": (
                    result.scores.format_compliance_score
                ),
                "missing_info_count": len(result.missing_information),
                "extraction_errors_count": len(result.extraction_errors),
                "improvement_suggestions_count": len(
                    result.improvement_suggestions
                ),
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
    """Define test cases for different synthesis ontology extractions."""

    return [
        {
            "name": "Complete_Hydrothermal_Synthesis",
            "target_material": "LiFePO4/C composite",
            "source_text": """
            Hydrothermal synthesis of LiFePO4/C composite was performed as 
            follows:
            2.66 g LiOH·H2O (99% purity, Sigma-Aldrich), 3.58 g FeSO4·7H2O 
            (analytical grade, Fisher Scientific), and 2.30 g NH4H2PO4 
            (ACS grade) were dissolved in 80 mL distilled water under magnetic 
            stirring at 500 rpm. Then 1.0 g glucose was added as carbon source 
            and the mixture was stirred for 30 minutes at room temperature. 
            The solution was transferred to a 100 mL Teflon-lined autoclave 
            (Parr Instruments) and heated at 180°C for 12 hours. 
            After cooling naturally to room temperature, the product was 
            filtered, washed with distilled water and ethanol, and dried at 
            80°C for 12 hours in an oven. Finally, the material was calcined 
            at 600°C for 2 hours under nitrogen atmosphere in a tube furnace.
            """,
            "extracted_ontology_json": json.dumps(
                {
                    "synthesis_id": "lifepo4_hydrothermal",
                    "target_compound": "LiFePO4/C composite",
                    "synthesis_method": "hydrothermal",
                    "starting_materials": [
                        {
                            "name": "LiOH·H2O",
                            "amount": 2.66,
                            "unit": "g",
                            "purity": "99%",
                            "vendor": "Sigma-Aldrich",
                        },
                        {
                            "name": "FeSO4·7H2O",
                            "amount": 3.58,
                            "unit": "g",
                            "purity": "analytical grade",
                            "vendor": "Fisher Scientific",
                        },
                        {
                            "name": "NH4H2PO4",
                            "amount": 2.30,
                            "unit": "g",
                            "purity": "ACS grade",
                        },
                        {
                            "name": "distilled water",
                            "amount": 80,
                            "unit": "mL",
                        },
                        {"name": "glucose", "amount": 1.0, "unit": "g"},
                    ],
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "dissolve",
                            "description": (
                                "Dissolve LiOH·H2O, FeSO4·7H2O, and NH4H2PO4 "
                                "in distilled water"
                            ),
                            "materials": [
                                {
                                    "name": "LiOH·H2O",
                                    "amount": 2.66,
                                    "unit": "g",
                                },
                                {
                                    "name": "FeSO4·7H2O",
                                    "amount": 3.58,
                                    "unit": "g",
                                },
                                {
                                    "name": "NH4H2PO4",
                                    "amount": 2.30,
                                    "unit": "g",
                                },
                                {
                                    "name": "distilled water",
                                    "amount": 80,
                                    "unit": "mL",
                                },
                            ],
                            "equipment": [{"name": "magnetic stirrer"}],
                            "conditions": {
                                "stirring": True,
                                "stirring_speed": 500,
                                "temperature": 25,
                                "temp_unit": "C",
                            },
                        },
                        {
                            "step_number": 2,
                            "action": "add",
                            "description": "Add glucose as carbon source",
                            "materials": [
                                {"name": "glucose", "amount": 1.0, "unit": "g"}
                            ],
                            "conditions": {
                                "duration": 30,
                                "time_unit": "min",
                                "stirring": True,
                            },
                        },
                        {
                            "step_number": 3,
                            "action": "heat",
                            "description": "Hydrothermal treatment",
                            "equipment": [
                                {
                                    "name": "Teflon-lined autoclave",
                                    "settings": "100 mL capacity",
                                    "instrument_vendor": "Parr Instruments",
                                }
                            ],
                            "conditions": {
                                "temperature": 180,
                                "temp_unit": "C",
                                "duration": 12,
                                "time_unit": "h",
                            },
                        },
                        {
                            "step_number": 4,
                            "action": "filter",
                            "description": "Filter and wash the product",
                        },
                        {
                            "step_number": 5,
                            "action": "dry",
                            "description": "Dry the product",
                            "equipment": [{"name": "oven"}],
                            "conditions": {
                                "temperature": 80,
                                "temp_unit": "C",
                                "duration": 12,
                                "time_unit": "h",
                            },
                        },
                        {
                            "step_number": 6,
                            "action": "calcine",
                            "description": "Calcination under nitrogen",
                            "equipment": [{"name": "tube furnace"}],
                            "conditions": {
                                "temperature": 600,
                                "temp_unit": "C",
                                "duration": 2,
                                "time_unit": "h",
                                "atmosphere": "N2",
                            },
                        },
                    ],
                    "equipment": [
                        {"name": "magnetic stirrer"},
                        {
                            "name": "Teflon-lined autoclave",
                            "instrument_vendor": "Parr Instruments",
                        },
                        {"name": "oven"},
                        {"name": "tube furnace"},
                    ],
                    "notes": (
                        "Glucose used as carbon source for composite formation"
                    ),
                },
                indent=2,
            ),
        },
        {
            "name": "Incomplete_Sol_Gel_Synthesis",
            "target_material": "TiO2 nanoparticles",
            "source_text": """
            Sol-gel synthesis of TiO2 nanoparticles: 10 mL titanium 
            tetraisopropoxide was added dropwise to 50 mL ethanol under 
            vigorous stirring. A mixture of 5 mL acetic acid and 10 mL 
            distilled water was added slowly. The solution was stirred for 2 
            hours at room temperature and aged for 24 hours. The gel was dried 
            at 100°C for 12 hours and calcined at 450°C for 3 hours with a 
            heating rate of 2°C/min.
            """,
            "extracted_ontology_json": json.dumps(
                {
                    "target_compound": "TiO2 nanoparticles",
                    "synthesis_method": "sol-gel",
                    "starting_materials": [
                        {
                            "name": "titanium tetraisopropoxide",
                            "amount": 10,
                            "unit": "mL",
                        },
                        {"name": "ethanol", "amount": 50, "unit": "mL"},
                    ],
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "add",
                            "description": (
                                "Add titanium tetraisopropoxide to ethanol"
                            ),
                            "conditions": {"stirring": True},
                        },
                        {
                            "step_number": 2,
                            "action": "heat",
                            "description": "Calcination",
                            "conditions": {
                                "temperature": 450,
                                "temp_unit": "C",
                                "duration": 3,
                                "time_unit": "h",
                            },
                        },
                    ],
                    "equipment": [{"name": "stirrer"}],
                },
                indent=2,
            ),
        },
        {
            "name": "Format_Error_Synthesis",
            "target_material": "Gold nanoparticles",
            "source_text": """
            Gold nanoparticles were synthesized using the Turkevich method. 
            250 mL of 1 mM HAuCl4 solution was heated to boiling. 25 mL of 
            38.8 mM sodium citrate solution was added quickly. The solution 
            was boiled for 15 minutes until the color changed to deep red.
            """,
            "extracted_ontology_json": json.dumps(
                {
                    "target_compound": "Gold nanoparticles",
                    "synthesis_method": "Turkevich method",
                    "starting_materials": [
                        {
                            "name": "HAuCl4",
                            "amount": "1 mM",  # Error: should be float with
                            # separate unit
                            "unit": "solution",  # Error: improper unit
                        },
                        {
                            "name": "sodium citrate",
                            "amount": "38.8 mM",  # Error: should be float
                            "volume": 25,  # Error: should be 'amount'
                        },
                    ],
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "heat",
                            "conditions": {
                                "temperature": "boiling",  # Error: should be
                                # numeric
                                "time": "15 minutes",  # Error: should be
                                # duration/time_unit
                            },
                        }
                    ],
                    "equipment": [],  # Missing equipment
                },
                indent=2,
            ),
        },
    ]


def print_evaluation_results(
    result: GeneralSynthesisEvaluation, test_name: str
):
    """Print detailed evaluation results in a formatted manner."""

    print(f"\n--- DETAILED RESULTS FOR {test_name} ---")
    print(f"Overall Score: {result.scores.overall_score}/5.0")
    print(f"Confidence Level: {result.confidence_level}")

    print("\n[Individual Scores]:")
    scores = result.scores
    print(
        f"  Structural Completeness: "
        f"{scores.structural_completeness_score}/5.0"
    )
    print(f"  Material Extraction: {scores.material_extraction_score}/5.0")
    print(f"  Process Steps: {scores.process_steps_score}/5.0")
    print(f"  Equipment Extraction: {scores.equipment_extraction_score}/5.0")
    print(f"  Conditions Extraction: {scores.conditions_extraction_score}/5.0")
    print(f"  Semantic Accuracy: {scores.semantic_accuracy_score}/5.0")
    print(f"  Format Compliance: {scores.format_compliance_score}/5.0")

    if result.missing_information:
        print(f"\n[Missing Information] ({len(result.missing_information)}):")
        for item in result.missing_information:
            print(f"  - {item}")

    if result.extraction_errors:
        print(f"\n[Extraction Errors] ({len(result.extraction_errors)}):")
        for error in result.extraction_errors:
            print(f"  - {error}")

    if result.improvement_suggestions:
        print(
            f"\n[Improvement Suggestions] "
            f"({len(result.improvement_suggestions)}):"
        )
        for suggestion in result.improvement_suggestions:
            print(f"  - {suggestion}")

    print("\n[High-level Reasoning]:")
    print(f"{result.reasoning}")


def save_individual_result(
    result: GeneralSynthesisEvaluation, test_name: str, case_num: int
):
    """Save individual evaluation result to JSON file."""

    current_dir = Path.cwd()
    filename = (
        current_dir / f"general_synthesis_evaluation_{case_num:02d}_"
        f"{test_name.lower()}.json"
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

    # Calculate statistics
    overall_scores = [r["overall_score"] for r in results]
    avg_score = sum(overall_scores) / len(overall_scores)
    print(f"Average overall score: {avg_score:.2f}")

    # Score distribution by confidence
    confidence_counts = {}
    for r in results:
        conf = r["confidence_level"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    print("\nScore distribution by confidence level:")
    for conf, count in confidence_counts.items():
        conf_scores = [
            r["overall_score"]
            for r in results
            if r["confidence_level"] == conf
        ]
        conf_avg = sum(conf_scores) / len(conf_scores) if conf_scores else 0
        print(f"  {conf}: {conf_avg:.2f} (n={count})")

    # Criterion averages
    print("\nAverage scores by criterion:")
    criterion_fields = [
        "structural_completeness_score",
        "material_extraction_score",
        "process_steps_score",
        "equipment_extraction_score",
        "conditions_extraction_score",
        "semantic_accuracy_score",
        "format_compliance_score",
    ]

    for field in criterion_fields:
        criterion_name = field.replace("_score", "").replace("_", " ").title()
        scores = [r[field] for r in results]
        avg = sum(scores) / len(scores)
        print(f"  {criterion_name}: {avg:.2f}")

    # Cases with issues
    print("\nTest cases with extraction issues:")
    issue_cases = [r for r in results if r["extraction_errors_count"] > 0]
    if issue_cases:
        for case in issue_cases:
            print(
                f"  {case['test_case']}: Score {case['overall_score']:.1f}, "
                f"Errors: {case['extraction_errors_count']}, "
                f"Missing: {case['missing_info_count']}"
            )
    else:
        print("  None")

    # Save statistics
    stats_filename = (
        current_dir / "general_synthesis_evaluation_statistics.json"
    )
    stats = {
        "total_cases": len(results),
        "average_overall_score": float(avg_score),
        "confidence_distribution": confidence_counts,
        "criterion_averages": {
            field.replace("_score", ""): float(
                sum(r[field] for r in results) / len(results)
            )
            for field in criterion_fields
        },
        "cases_with_issues": len(issue_cases),
        "detailed_results": results,
    }

    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDetailed statistics saved to: {stats_filename}")

    # Generate CSV
    summary_filename = current_dir / "general_synthesis_evaluation_summary.csv"
    with open(summary_filename, "w") as f:
        if results:
            header = ",".join(results[0].keys())
            f.write(header + "\n")
            for result in results:
                row = ",".join(str(result[key]) for key in result.keys())
                f.write(row + "\n")

    print(f"Summary CSV saved to: {summary_filename}")
    print(f"\nAll results saved in: {current_dir}")


if __name__ == "__main__":
    main()
