from abc import ABC, abstractmethod

from llm_synthesis.models.paper import Paper


class PaperLoaderInterface(ABC):
    """
    Interface for a paper loader that returns a list of papers.
    """

    @abstractmethod
    def load(self) -> list[Paper]:
        pass
