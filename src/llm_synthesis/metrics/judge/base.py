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

from llm_synthesis.metrics.judge.evaluation_ontology import SynthesisEvaluation

T = TypeVar("T")
R = TypeVar("R")


class JudgeMeta(ProgramMeta, ABCMeta):
    pass


class JudgeInterface(dspy.Module, Generic[T, R], metaclass=JudgeMeta):
    """
    Generic interface for a judge that takes an input of type T
    and returns an evaluation of type R.
    """

    @abstractmethod
    def forward(self, input: T) -> R:
        pass


# Input is a tuple of (target_material, extracted_recipe, synthesis_procedure)
SynthesisJudgeInterface = JudgeInterface[
    tuple[str, str, str], SynthesisEvaluation
]
