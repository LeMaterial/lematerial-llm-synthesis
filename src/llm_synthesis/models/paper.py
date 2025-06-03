from pydantic import BaseModel

from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class Paper(BaseModel):
    publication_text: str
    si_text: str = ""


class PaperWithSynthesisParagraph(Paper):
    synthesis_paragraph: str


class PaperWithSynthesisOntology(Paper):
    synthesis_ontology: GeneralSynthesisOntology
