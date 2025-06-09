from abc import abstractmethod
from typing import TypeVar

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.transformers.base import ExtractorInterface

R = TypeVar("R")


class StructuredSynthesisExtractorInterface(
    ExtractorInterface[str, GeneralSynthesisOntology]
):
    """
    This interface is used to extract a structured synthesis from a synthesis paragraph.
    """

    @abstractmethod
    def extract(self, input: str) -> GeneralSynthesisOntology:
        """
        Extract a structured synthesis ontology from the given synthesis paragraph.

        Args:
            input (str): The synthesis paragraph to process.

        Returns:
            GeneralSynthesisOntology: The structured synthesis ontology extracted from the paragraph.
        """
        pass
