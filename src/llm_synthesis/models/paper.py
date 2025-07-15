from pydantic import BaseModel

from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class SynthesisEntry(BaseModel):
    material: str
    synthesis: GeneralSynthesisOntology | None = None


class Paper(BaseModel):
    name: str
    id: str
    publication_text: str
    si_text: str = ""


class PaperWithSynthesisOntologies(Paper):
    all_syntheses: list[SynthesisEntry]
