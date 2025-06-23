from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class ResultGatherInterface(ABC, Generic[T]):
    @abstractmethod
    def gather(self, paper: T):
        """
        Gather the results of the paper.
        """
        pass
