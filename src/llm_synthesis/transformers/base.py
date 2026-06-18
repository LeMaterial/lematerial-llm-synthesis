import asyncio
from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import dspy

# dspy >= 3.x moved ProgramMeta from dspy.primitives.program to
# dspy.primitives.module. Try the new location first, fall back for older
# installs.
try:
    from dspy.primitives.module import ProgramMeta
except ImportError:  # pragma: no cover - dspy < 3.x
    from dspy.primitives.program import ProgramMeta

T = TypeVar("T")
R = TypeVar("R")


class ExtractorMeta(ProgramMeta, ABCMeta):
    pass


class ExtractorInterface(dspy.Module, Generic[T, R], metaclass=ExtractorMeta):
    """
    Generic interface for an extractor that takes an input of type T
    and returns an output of type R.
    """

    @abstractmethod
    def forward(self, input: T) -> R:
        pass

    async def aforward(self, input: T) -> R:
        return await asyncio.to_thread(self.forward, input)
