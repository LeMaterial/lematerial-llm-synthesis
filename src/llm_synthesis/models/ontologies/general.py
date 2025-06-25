"""General synthesis ontology."""

from typing import Literal

from pydantic import BaseModel, Field


class Material(BaseModel):
    vendor: str | None = Field(
        ...,
        description="Vendor of the material."
        " E.g. 'Sinopharm Chemical Reagent Co. Ltd.'.",
    )
    name: str = Field(
        ...,
        description="Name of the material."
        " E.g. 'Nickel Nitrate', 'Cobalt Nitrate',"
        " 'Deionized Water', 'Ammonia Solution'.",
    )
    amount: float = Field(
        ...,
        description="Amount of material used in the synthesis."
        " Just the number, no unit.",
    )
    unit: str = Field(
        ..., description="Unit of the amount. E.g. 'g', 'mol', 'wt%'."
    )
    role: Literal["precursor", "support", "solvent", "additive", "reagent"]
    stoichiometry: str | None = Field(
        ...,
        description="Stoichiometry of the material in the synthesis."
        " E.g. '1:1', '1:2', '2:1'.",
    )


class Conditions(BaseModel):
    temperature: float | None = Field(
        ..., description="Temperature of the synthesis. E.g. 100, 200, 300."
    )
    temp_unit: str | None = Field(
        ..., description="Unit of the temperature. E.g. 'C', 'K'."
    )
    duration: float | None = Field(
        ..., description="Duration of the synthesis. E.g. 1, 2, 3."
    )
    time_unit: str | None = Field(
        ..., description="Unit of the duration. E.g. 'h', 'min', 's'."
    )
    atmosphere: str | None = Field(
        ..., description="Atmosphere of the synthesis. E.g. 'air', 'N2', 'H2'."
    )
    stirring: bool | None = Field(
        ..., description="Whether the synthesis is stirred."
    )


class ProcessStep(BaseModel):
    action: Literal[
        "add",
        "mix",
        "heat",
        "reflux",
        "age",
        "filter",
        "wash",
        "dry",
        "reduce",
        "calcine",
    ]
    description: str | None = Field(
        ..., description="Description of the process step."
    )
    materials: list[Material] = Field(
        ..., description="Materials used in the process step."
    )
    conditions: Conditions | None = Field(
        ..., description="Conditions of the process step."
    )


class GeneralSynthesisOntology(BaseModel):
    target_compound: str = Field(
        ..., description="Target compound composition."
    )
    starting_materials: list[str] = Field(
        ..., description="Starting materials used in the synthesis."
    )
    steps: list[ProcessStep] = Field(
        ..., description="Process steps of the synthesis."
    )
    notes: str | None = Field(..., description="Notes about the synthesis.")

    def keys(self):
        return self.model_dump().keys()

    def __getitem__(self, key: str):
        return self.model_dump()[key]
