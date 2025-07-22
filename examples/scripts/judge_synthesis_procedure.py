import json
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Instantiate the judge from the Hydra config
    judge: SynthesisJudgeInterface = instantiate(cfg.judge.architecture)
    log.info(f"Instantiated judge: {type(judge).__name__}")

    # --- Example Data ---
    target_material = "Bi1.5Sb0.5Te1.7Se1.3 (BSTS)23-27 ultrathin films"

    synthesis_procedure = (
        "BSTS thin films with 10 nm film in thickness are prepared on mica by "
        "van der Waals epitaxy physical vapor deposition. We employed the "
        "Bi1.5Sb0.5Te1.7Se1.3 composition as the source material for the thin "
        "film crystal growth, where the surface state is tuned to n-type "
        "under the insulating bulk state. F4-TCNQ was deposited on the half "
        "of a BSTS thin film by thermal evaporation under the pressure of "
        "10^-6 Pa to form the p-n junction on the top surface of BSTS thin "
        "films."
    )

    extracted_recipe = """
   {
    "id": "BSTS_thin_films_synthesis",
    "target_compound": "BSTS thin films",
    "materials": [
        "Bi1.5Sb0.5Te1.7Se1.3",
        "F4-TCNQ",
        "mica"
    ],
    "steps": [
        {
            "action": "add",
            "materials": [
                "Bi1.5Sb0.5Te1.7Se1.3"
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": "vacuum",
                "stirring": null
            }
        },
        {
            "action": "heat",
            "materials": [
                "BSTS thin films"
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": "10^-6 Pa",
                "stirring": null
            }
        },
        {
            "action": "add",
            "materials": [
                "F4-TCNQ"
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": "10^-6 Pa",
                "stirring": null
            }
        }
    ],
    "notes": (
        "BSTS thin films are prepared on mica by van der Waals epitaxy "
        "physical vapor deposition to form a p-n junction."
    )
    """

    # Optional synthesis context for enhanced judges
    synthesis_context = (
        "Material class: Semiconductor; Application domain: Electronics; "
        "Synthesis method: Physical vapor deposition"
    )

    log.info("--- Evaluating Recipe ---")
    log.info(f"Target: {target_material}")
    log.info(f"Synthesis Procedure: {synthesis_procedure}")
    log.info(f"Extracted: {extracted_recipe}")
    log.info("------------------------")

    # Check if judge supports 4-tuple input (enhanced) or only 3-tuple (basic)
    try:
        # Try enhanced input with context
        evaluation_input = (
            target_material,
            extracted_recipe,
            synthesis_procedure,
            synthesis_context,
        )
        result = judge.forward(evaluation_input)
        log.info("Using enhanced judge with synthesis context")
    except (TypeError, ValueError) as e:
        # Fall back to basic 3-tuple input
        log.info(f"Falling back to basic judge (3-tuple input): {e}")
        evaluation_input = (
            target_material,
            extracted_recipe,
            synthesis_procedure,
        )
        result = judge.forward(evaluation_input)

    # Print the results
    log.info("\n--- JUDGE'S EVALUATION ---")
    log.info("\n[Reasoning]:")
    print(result.reasoning)

    # Check if this is an enhanced evaluation with structured scores
    if hasattr(result, "scores") and hasattr(
        result.scores, "materials_appropriateness_score"
    ):
        log.info("\n[Synthesis Evaluation Scores]:")
        print(json.dumps(result.scores.model_dump(), indent=2))

        # Print additional enhanced features
        if hasattr(result, "confidence_level"):
            log.info(f"\n[Confidence Level]: {result.confidence_level}")

        if hasattr(result, "critical_issues") and result.critical_issues:
            log.info("\n[Critical Issues]:")
            for issue in result.critical_issues:
                print(f"- {issue}")

        if hasattr(result, "recommendations") and result.recommendations:
            log.info("\n[Recommendations]:")
            for rec in result.recommendations:
                print(f"- {rec}")
    else:
        # Basic evaluation format
        log.info("\n[Basic Scores]:")
        print(json.dumps(result.scores.model_dump(), indent=2))

    log.info("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
