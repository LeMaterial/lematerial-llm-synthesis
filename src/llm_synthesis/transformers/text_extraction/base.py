from abc import abstractmethod

from llm_synthesis.transformers.base import ExtractorInterface


class TextExtractorInterface(ExtractorInterface[str, str]):
    """
    Interface for a text extractor that takes a str and returns a string extracted from the paper str.
    """

    @abstractmethod
    def extract(self, input: str) -> str:
        """
        Extract text from the given paper.

        Args:
            input (str): The paper text from which to extract text.

        Returns:
            str: The extracted text from the paper.
        """
        pass
