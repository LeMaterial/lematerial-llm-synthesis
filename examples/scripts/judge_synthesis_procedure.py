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

    synthesis_procedure = """BSTS thin films with 10 nm film in thickness are prepared on mica by van der Waals epitaxy physical vapor deposition. We employed the Bi1.5Sb0.5Te1.7Se1.3 composition as the source material for the thin film crystal growth, where the surface state is tuned to n-type under the insulating bulk state. F4-TCNQ was deposited on the half of a BSTS thin film by thermal evaporation under the pressure of 10^-6 Pa to form the p-n junction on the top surface of BSTS thin films. """
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
    "notes": "BSTS thin films are prepared on mica by van der Waals epitaxy physical vapor deposition to form a p-n junction."
    """

    log.info("--- Evaluating Recipe ---")
    log.info(f"Target: {target_material}")
    log.info(f"Synthesis Procedure: {synthesis_procedure}")
    log.info(f"Extracted: {extracted_recipe}")
    log.info("------------------------")

    # Run the evaluation
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

    log.info("\n[Scores]:")
    print(json.dumps(result.scores.model_dump(), indent=2))

    log.info("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main() 