from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import dspy

T = TypeVar("T")
R = TypeVar("R")


class ExtractorInterface(dspy.Module, Generic[T, R], metaclass=ABCMeta):
    """
    Generic interface for an extractor that takes an input of type T and returns
    an output of type R.
    """

    @abstractmethod
    def forward(self, input: T) -> R:
        pass
