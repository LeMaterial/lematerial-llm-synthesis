"""General synthesis ontology."""

from typing import Literal

from pydantic import BaseModel, Field


class Material(BaseModel):
    vendor: str | None = Field(
        default=None,
        description=(
            "Vendor of the material. E.g. 'Sinopharm Chemical Reagent Co. "
            "Ltd.'."
        ),
    )
    name: str = Field(
        ...,
        description=(
            "Name of the material. E.g. 'Nickel Nitrate', 'Cobalt Nitrate', "
            "'Deionized Water', 'Ammonia Solution'."
        ),
    )
    amount: float | None = Field(
        default=None,
        description=(
            "Amount of material used in the synthesis. Just the number, "
            "no unit."
        ),
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the amount. E.g. 'g', 'mol', 'wt%', 'mL'.",
    )
    purity: str | None = Field(
        default=None,
        description=(
            "Purity of the material. E.g. '99%', 'ACS grade', "
            "'analytical grade'."
        ),
    )
    role: Literal[
        "precursor",
        "support",
        "solvent",
        "additive",
        "reagent",
        "catalyst",
        "reductant",
        "oxidant",
    ] = Field(..., description="Role of the material in the synthesis.")
    stoichiometry: str | None = Field(
        default=None,
        description=(
            "Stoichiometry of the material in the synthesis. E.g. '1:1', "
            "'1:2', '2:1'."
        ),
    )


class Equipment(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Name of the equipment. E.g. 'autoclave', 'tube furnace', "
            "'magnetic stirrer'."
        ),
    )
    specifications: str | None = Field(
        default=None,
        description=(
            "Specifications of the equipment. E.g. '100 mL Teflon-lined', "
            "'max 1200°C'."
        ),
    )
    settings: str | None = Field(
        default=None,
        description=(
            "Operating settings. E.g. '500 rpm', 'heating rate 5°C/min'."
        ),
    )


class Conditions(BaseModel):
    temperature: float | None = Field(
        default=None,
        description="Temperature of the synthesis. E.g. 100, 200, 300.",
    )
    temp_unit: str | None = Field(
        default=None,
        description="Unit of the temperature. E.g. 'C', 'K', 'F'.",
    )
    duration: float | None = Field(
        default=None, description="Duration of the synthesis. E.g. 1, 2, 3."
    )
    time_unit: str | None = Field(
        default=None,
        description="Unit of the duration. E.g. 'h', 'min', 's', 'days'.",
    )
    pressure: float | None = Field(
        default=None, description="Pressure of the synthesis. E.g. 1, 10, 100."
    )
    pressure_unit: str | None = Field(
        default=None,
        description="Unit of pressure. E.g. 'atm', 'bar', 'Pa', 'torr'.",
    )
    atmosphere: str | None = Field(
        default=None,
        description=(
            "Atmosphere of the synthesis. E.g. 'air', 'N2', 'H2', 'vacuum'."
        ),
    )
    stirring: bool | None = Field(
        default=None, description="Whether the synthesis is stirred."
    )
    stirring_speed: float | None = Field(
        default=None, description="Stirring speed in rpm."
    )
    ph: float | None = Field(
        default=None, description="pH of the solution. E.g. 7.0, 8.5, 12.0."
    )


class ProcessStep(BaseModel):
    step_number: int = Field(
        ..., description="Sequential step number in the synthesis procedure."
    )
    action: Literal[
        "add",
        "mix",
        "heat",
        "cool",
        "reflux",
        "age",
        "filter",
        "wash",
        "dry",
        "reduce",
        "calcine",
        "dissolve",
        "precipitate",
        "centrifuge",
        "sonicate",
        "anneal",
    ] = Field(..., description="Primary action performed in this step.")
    description: str | None = Field(
        default=None, description="Detailed description of the process step."
    )
    materials: list[Material] = Field(
        default_factory=list, description="Materials used in the process step."
    )
    equipment: list[Equipment] = Field(
        default_factory=list, description="Equipment used in the process step."
    )
    conditions: Conditions | None = Field(
        default=None, description="Conditions of the process step."
    )
    safety_notes: str | None = Field(
        default=None, description="Safety considerations for this step."
    )


class CharacterizationMethod(BaseModel):
    technique: str = Field(
        ...,
        description=(
            "Characterization technique. E.g. 'XRD', 'SEM', 'TEM', 'XPS', "
            "'BET', 'IR'."
        ),
    )
    purpose: str | None = Field(
        default=None,
        description=(
            "Purpose of the characterization. E.g. 'crystal structure', "
            "'morphology', 'composition'."
        ),
    )
    conditions: str | None = Field(
        default=None,
        description=(
            "Characterization conditions. E.g. 'Cu K-alpha radiation', "
            "'20 kV acceleration voltage'."
        ),
    )


class YieldInformation(BaseModel):
    amount: float | None = Field(
        default=None, description="Amount of product obtained."
    )
    unit: str | None = Field(
        default=None, description="Unit of the yield amount. E.g. 'g', 'mg'."
    )
    percentage: float | None = Field(
        default=None,
        description="Percentage yield based on theoretical yield.",
    )


class GeneralSynthesisOntology(BaseModel):
    """
    Comprehensive synthesis ontology for structured synthesis procedures.
    """

    # Basic Information
    synthesis_id: str | None = Field(
        default=None,
        description="Unique identifier for the synthesis procedure.",
    )
    target_compound: str = Field(
        ..., description="Target compound composition and description."
    )
    synthesis_method: str | None = Field(
        default=None,
        description=(
            "Overall synthesis method. E.g. 'hydrothermal', 'sol-gel', "
            "'solid-state'."
        ),
    )

    # Materials
    starting_materials: list[Material] = Field(
        default_factory=list,
        description="All starting materials used in the synthesis.",
    )

    # Procedure
    steps: list[ProcessStep] = Field(
        default_factory=list,
        description="Sequential process steps of the synthesis.",
    )

    # Equipment and Safety
    major_equipment: list[Equipment] = Field(
        default_factory=list,
        description="Major equipment used throughout the synthesis.",
    )
    safety_considerations: str | None = Field(
        default=None,
        description="Overall safety considerations and precautions.",
    )

    # Characterization and Results
    characterization_methods: list[CharacterizationMethod] = Field(
        default_factory=list,
        description="Methods used to characterize the final product.",
    )
    yield_information: YieldInformation | None = Field(
        default=None, description="Information about product yield."
    )

    # Additional Information
    notes: str | None = Field(
        default=None, description="Additional notes about the synthesis."
    )
    references: str | None = Field(
        default=None,
        description="Literature references or source information.",
    )
