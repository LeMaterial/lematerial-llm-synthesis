from typing import TypeVar

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.transformers.base import ExtractorInterface

R = TypeVar("R")


StructuredSynthesisExtractorInterface = ExtractorInterface[
    str, GeneralSynthesisOntology
]
