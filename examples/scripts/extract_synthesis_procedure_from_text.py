import logging
import os
import json

import hydra
from hydra.utils import instantiate, get_original_cwd
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
    # Conditional instantiation to support both nested (default) and flat (local) configs
    if hasattr(cfg.data_loader, 'architecture'):
        data_loader_config = cfg.data_loader.architecture
    else:
        data_loader_config = cfg.data_loader

    # Apply absolute path resolution only if data_dir exists in the selected configuration
    # This is necessary because chdir: true changes the current working directory
    if hasattr(data_loader_config, 'data_dir'):
        original_cwd = get_original_cwd()
        data_loader_config.data_dir = os.path.join(original_cwd, data_loader_config.data_dir)

    data_loader: PaperLoaderInterface = instantiate(
        data_loader_config
    )
    papers = data_loader.load()

    # Resolve prompt_path for synthesis_extraction to be an absolute path
    # This is necessary because chdir: true changes the current working directory
    if hasattr(cfg.synthesis_extraction.architecture.lm.system_prompt, "prompt_path"):
        original_cwd = get_original_cwd()
        cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path = os.path.join(
            original_cwd, cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path
        )

    paragraph_extractor: TextExtractorInterface = instantiate(
        cfg.paragraph_extraction.architecture
    )
    synthesis_extractor: StructuredSynthesisExtractorInterface = instantiate(
        cfg.synthesis_extraction.architecture
    )

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
