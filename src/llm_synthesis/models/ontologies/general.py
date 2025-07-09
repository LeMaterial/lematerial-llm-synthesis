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
            "Amount of material used in the synthesis. Just the number, no "
            "unit. Optional for cases where materials are used in excess or "
            "'until completion'."
        ),
    )
    unit: str | None = Field(
        default=None,
        description=(
            "Unit of the amount. Examples by category: "
            "Mass: 'g', 'mg', 'μg', 'kg'; "
            "Volume: 'mL', 'μL', 'L', 'drops'; "
            "Molar: 'mol', 'mmol', 'μmol'; "
            "Concentration: 'M', 'mM', 'μM', 'm' (molality), 'N' (normality); "
            "Percentage: 'wt%', 'mol%', 'at%', 'vol%', 'w/w', 'w/v', 'v/v'; "
            "Parts: 'ppm', 'ppb'; "
            "Pressure: 'atm', 'bar', 'Pa', 'kPa', 'MPa', 'torr', 'psi'; "
            "Electrochemical: 'C', 'mAh', 'V', 'mV', 'A', 'mA'; "
            "Equivalents: 'equiv', 'meq'; "
            "Descriptive: 'excess', 'stoichiometric', 'catalytic amount', "
            "'trace'."
        ),
    )
    purity: str | None = Field(
        default=None,
        description=(
            "Purity of the material. E.g. '99%', '99.9%', 'ACS grade', "
            "'analytical grade', 'technical grade', 'reagent grade'."
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
    instrument_vendor: str | None = Field(
        default=None,
        description=(
            "Vendor of the instrument. E.g. 'Thermo Fisher Scientific', "
            "'Agilent Technologies', 'Bruker', 'PerkinElmer', 'Shimadzu'."
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
        description=(
            "Unit of pressure. E.g. 'atm', 'bar', 'Pa', 'torr', 'psi'."
        ),
    )
    atmosphere: str | None = Field(
        default=None,
        description=(
            "Atmosphere of the synthesis. E.g. 'air', 'N2', 'H2', 'Ar', "
            "'O2', 'vacuum'."
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
        "ion exchange",
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


class CharacterizationMethod(BaseModel):
    technique: str = Field(
        ...,
        description=(
            "Characterization technique. Use standard acronyms where possible "
            "E.g. 'XRD', 'SEM', 'TEM', 'XPS', 'BET', 'FTIR', 'UV-Vis', 'NMR', "
            "'X-ray diffraction', 'scanning electron microscopy'."
        ),
    )
    purpose: str | None = Field(
        default=None,
        description=(
            "Purpose of the characterization. E.g. 'crystal structure', "
            "'morphology', 'composition', 'surface area', "
            "'phase identification' "
        ),
    )
    conditions: str | None = Field(
        default=None,
        description=(
            "Characterization conditions. E.g. 'Cu K-alpha radiation', "
            "'20 kV acceleration voltage', '2θ range 10-80°'."
        ),
    )


class GeneralSynthesisOntology(BaseModel):
    """
    Comprehensive synthesis ontology for structured synthesis procedures.
    """

    synthesis_id: str | None = Field(
        default=None,
        description=(
            "Unique identifier for the synthesis procedure within a paper. "
            "E.g. 'synthesis_1', 'method_A', 'sample_LFP'. Useful for papers "
            "with multiple synthesis protocols."
        ),
    )
    target_compound: str = Field(
        ..., description="Target compound composition and description."
    )
    synthesis_method: str | None = Field(
        default=None,
        description=(
            "Overall synthesis method. E.g. 'hydrothermal', 'sol-gel', "
            "'solid-state', 'chemical vapor deposition', 'electrodeposition'."
        ),
    )

    starting_materials: list[Material] = Field(
        default_factory=list,
        description=(
            "All starting materials and precursors used in the synthesis."
        ),
    )

    steps: list[ProcessStep] = Field(
        default_factory=list,
        description="Sequential process steps of the synthesis.",
    )

    equipment: list[Equipment] = Field(
        default_factory=list,
        description="Major equipment used throughout the synthesis.",
    )

    notes: str | None = Field(
        default=None,
        description=(
            "Additional notes about the synthesis procedure, important "
            "observations, or variations mentioned in the text."
        ),
    )

    def keys(self):
        return self.model_dump().keys()

    def __getitem__(self, key: str):
        return self.model_dump()[key]
