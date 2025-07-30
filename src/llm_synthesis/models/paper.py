from typing import Any

from pydantic import BaseModel

from llm_synthesis.metrics.judge import GeneralSynthesisEvaluation
from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.models.plot import ExtractedLinePlotData, ExtractedPlotData


class ImageData(BaseModel):
    """Image data structure from HuggingFace datasets."""

    bytes: bytes
    path: str


class SynthesisEntry(BaseModel):
    material: str
    synthesis: GeneralSynthesisOntology | None = None
    evaluation: GeneralSynthesisEvaluation | None = None


class Paper(BaseModel):
    name: str
    id: str
    publication_text: str
    si_text: str = ""
    images: list[ImageData] | None = None


class PaperWithSynthesisOntologiesAndFigures(Paper):
    all_syntheses: list[SynthesisEntry]
    cost_data: dict[str, Any] | None = None
    figures: list[FigureInfo]
    extracted_data_from_figures: list[
        ExtractedLinePlotData | ExtractedPlotData | str | None
    ]
