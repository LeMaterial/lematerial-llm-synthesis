import json
import logging
import os

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
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
    data_loader: PaperLoaderInterface = instantiate(
        cfg.data_loader.architecture
    )
    papers = data_loader.load()

    if hasattr(
        cfg.synthesis_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        original_cwd = get_original_cwd()
        prompt_path = os.path.join(
            original_cwd,
            cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path,
        )
        cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path = (
            prompt_path
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

        os.makedirs(paper.id, exist_ok=True)

        with open(os.path.join(paper.id, "result.json"), "w") as f:
            f.write(
                json.dumps(
                    structured_synthesis_procedure.model_dump(), indent=2
                )
            )

        with open(os.path.join(paper.id, "synthesis_paragraph.txt"), "w") as f:
            f.write(synthesis_paragraph)

        with open(os.path.join(paper.id, "publication_text.txt"), "w") as f:
            f.write(paper.publication_text)

        with open(os.path.join(paper.id, "si_text.txt"), "w") as f:
            f.write(paper.si_text)

    logging.info("Success")


if __name__ == "__main__":
    main()
