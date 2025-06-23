import json
import logging
import os

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.paper import PaperWithSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    StructuredSynthesisExtractorInterface,
)
from llm_synthesis.transformers.text_extraction.base import (
    TextExtractorInterface,
)
from llm_synthesis.utils import remove_figs


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    # Ensure data directory is correctly set if it's defined in the config
    if hasattr(cfg.data_loader.architecture, "data_dir"):
        cfg.data_loader.architecture.data_dir = os.path.join(
            original_cwd, cfg.data_loader.architecture.data_dir
        )

    # Load data
    data_loader: PaperLoaderInterface = instantiate(cfg.data_loader.architecture)
    papers = data_loader.load()

    # Handle system prompt path if defined
    if hasattr(
        cfg.synthesis_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        prompt_path = os.path.join(
            original_cwd,
            cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path,
        )
        cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path = (
            prompt_path
        )

    # Initialize text and synthesis extractors
    paragraph_extractor: TextExtractorInterface = instantiate(
        cfg.paragraph_extraction.architecture
    )
    synthesis_extractor: StructuredSynthesisExtractorInterface = instantiate(
        cfg.synthesis_extraction.architecture
    )

    # Process each paper
    for paper in papers:
        logging.info(f"Processing {paper.name}")
        synthesis_paragraph = paragraph_extractor.forward(
            input=remove_figs(
                paper.publication_text
            ),  # Removing figures avoid token overload
        )

        os.makedirs(paper.id, exist_ok=True)

        structured_synthesis_procedure = synthesis_extractor.forward(
            input=synthesis_paragraph,
        )
        logging.info(structured_synthesis_procedure)
        paper_enriched = PaperWithSynthesisOntology(
            **paper.model_dump(),
            synthesis_paragraph=synthesis_paragraph,
            synthesis_ontology=structured_synthesis_procedure,
        )

        os.makedirs(paper.id, exist_ok=True)

        with open(os.path.join(paper.id, "result.json"), "w") as f:
            f.write(json.dumps(paper_enriched.model_dump(), indent=2))

    logging.info("Success")


if __name__ == "__main__":
    main()