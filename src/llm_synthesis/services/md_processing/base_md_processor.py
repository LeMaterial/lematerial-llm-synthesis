from abc import ABC, abstractmethod
from typing import Optional


class BaseMarkdownProcessor(ABC):
    @abstractmethod
    def process_markdown(
        self, markdown_data: str, extra_markdown_data: Optional[str]
    ) -> str:
        """
        Processes the given markdown data and returns the processed markdown.

        Args:
            markdown_data: The markdown data as a string.

        Returns:
            The processed markdown data as a string.
        """
        pass
