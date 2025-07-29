from typing import Any

from pydantic import BaseModel

from llm_synthesis.metrics.judge import GeneralSynthesisEvaluation
from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class SynthesisEntry(BaseModel):
    material: str
    synthesis: GeneralSynthesisOntology | None = None
    evaluation: GeneralSynthesisEvaluation | None = None


class Paper(BaseModel):
    name: str
    id: str
    publication_text: str
    si_text: str = ""


class PaperWithSynthesisOntologiesAndFigures(Paper):
    all_syntheses: list[SynthesisEntry]
    cost_data: dict[str, Any] | None = None
    figures: list[FigureInfo]
