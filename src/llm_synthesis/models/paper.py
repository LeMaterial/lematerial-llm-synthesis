from pydantic import BaseModel

from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class Paper(BaseModel):
    name: str
    id: str
    publication_text: str
    si_text: str = ""


class PaperWithSynthesisParagraph(Paper):
    synthesis_paragraph: str


class PaperWithSynthesisOntology(PaperWithSynthesisParagraph):
    synthesis_ontology: GeneralSynthesisOntology
