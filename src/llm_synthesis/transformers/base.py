from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ExtractorInterface(ABC, Generic[T, R]):
    """
    Generic interface for an extractor that takes an input of type T and returns an output of type R.
    """

    @abstractmethod
    def extract(self, input: T) -> R:
        pass
