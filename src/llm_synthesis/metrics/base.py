from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class MetricInterface(ABC, Generic[T]):
    """
    Generic interface for a metric that takes two inputs of type T
    and returns a float.
    """

    @abstractmethod
    def __call__(self, preds: T, refs: T) -> float:
        pass
