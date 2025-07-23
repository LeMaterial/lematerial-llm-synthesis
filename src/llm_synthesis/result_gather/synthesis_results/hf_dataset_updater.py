import logging
from typing import Any

from llm_synthesis.models.paper import PaperWithSynthesisOntologies
from llm_synthesis.result_gather.base import ResultGatherInterface

logger = logging.getLogger(__name__)


class HFSynthesisUpdater(ResultGatherInterface[PaperWithSynthesisOntologies]):
    """
    A ResultGatherInterface that collects synthesis results in memory
    for later batch updates of Hugging Face Datasets.
    """

    def __init__(self, update_column: str = "structured_synthesis"):
        self._gathered_syntheses: dict[str, list[dict[str, Any]]] = {}
        self.update_column = update_column
        logger.info(
            f"Initialized HFSynthesisUpdater to update column"
            f"'{self.update_column}'."
        )

    def gather(self, paper: PaperWithSynthesisOntologies):
        """
        Gathers the synthesis results for a
        single paper and stores them internally.
        """
        paper_id = str(paper.id)

        structured_synthesis_data = []
        for entry in paper.all_syntheses:
            if entry.synthesis:
                structured_synthesis_data.append(
                    {
                        "material": entry.material,
                        "synthesis": entry.synthesis.model_dump(),
                    }
                )
            else:
                structured_synthesis_data.append(
                    {"material": entry.material, "synthesis": None}
                )

        self._gathered_syntheses[paper_id] = structured_synthesis_data
        logger.debug(f"Gathered synthesis for paper ID: {paper_id}")

    def get_accumulated_synthesis_data(self) -> dict[str, list[dict[str, Any]]]:
        """
        Returns the accumulated synthesis data, keyed by paper ID.
        """
        return self._gathered_syntheses

    def get_update_column_name(self) -> str:
        """
        Returns the name of the column to be updated in the dataset.
        """
        return self.update_column
