from pydantic import BaseModel

from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class Paper(BaseModel):
    name: str
    id: str
    publication_text: str
    si_text: str = ""


class PaperWithSynthesisOntology(Paper):
    synthesis_ontology: GeneralSynthesisOntology
