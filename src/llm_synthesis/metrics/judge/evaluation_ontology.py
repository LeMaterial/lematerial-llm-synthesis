# Contains an ontology for the evaluation of a synthesis recipe.
# Based on the following paper:
# https://arxiv.org/html/2502.16457v1

from pydantic import BaseModel, Field


class SynthesisEvaluationScore(BaseModel):
    """
    A structured model for scoring a synthesis recipe based on multiple
    criteria. Scores are on a scale of 1.0 (poor) to 5.0 (excellent).
    """

    materials_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the appropriateness of selected materials."
        ),
        ge=1.0,
        le=5.0,
    )
    materials_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for the materials appropriateness score."
        ),
    )

    equipment_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the appropriateness of the selected equipment."
        ),
        ge=1.0,
        le=5.0,
    )
    equipment_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for the equipment appropriateness score."
        ),
    )

    procedure_completeness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the completeness and detail of the procedure."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_completeness_reasoning: str = Field(
        ...,
        description="Detailed reasoning for the procedure completeness score.",
    )

    procedure_similarity_score: float = Field(
        ...,
        description=(
            "Score (1-5) for how closely the procedure matches the ground "
            "truth."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_similarity_reasoning: str = Field(
        ...,
        description="Detailed reasoning for the procedure similarity score.",
    )

    procedure_feasibility_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the realistic feasibility of the procedure in "
            "a lab."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_feasibility_reasoning: str = Field(
        ...,
        description="Detailed reasoning for the procedure feasibility score.",
    )

    characterization_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the appropriateness of characterization metrics."
        ),
        ge=1.0,
        le=5.0,
    )
    characterization_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for the characterization appropriateness "
            "score."
        ),
    )

    characterization_similarity_score: float = Field(
        ...,
        description=(
            "Score (1-5) for how well predicted properties match actual "
            "results."
        ),
        ge=1.0,
        le=5.0,
    )
    characterization_similarity_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for the characterization similarity score."
        ),
    )

    overall_score: float = Field(
        ...,
        description=(
            "The average of all other scores, representing the overall "
            "quality."
        ),
        ge=1.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        ...,
        description=(
            "Overall reasoning summarizing the evaluation and final "
            "assessment."
        ),
    )


class SynthesisEvaluation(BaseModel):
    """
    Represents a complete evaluation of a synthesis recipe, including
    step-by-step reasoning and a structured score object.
    """

    reasoning: str = Field(
        ...,
        description="High-level reasoning overview for the evaluation.",
    )
    scores: SynthesisEvaluationScore = Field(
        ...,
        description=(
            "The structured JSON object containing all scores and individual "
            "reasoning."
        ),
    )
