"""AlchemyBench evaluation ontology for enhanced synthesis recipe
assessment."""

from typing import Literal

from pydantic import BaseModel, Field


class AlchemyBenchSynthesisEvaluationScore(BaseModel):
    """
    Enhanced evaluation scores following AlchemyBench methodology.
    Scores are on a scale of 1.0 (poor) to 5.0 (excellent) with 0.5 increments.
    """

    # Materials Assessment
    materials_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the appropriateness and accuracy of selected "
            "materials, including quantities, purities, and stoichiometries."
        ),
        ge=1.0,
        le=5.0,
    )
    materials_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning covering material selection, quantities, "
            "chemical compatibility, and availability."
        ),
    )

    # Equipment Assessment
    equipment_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the suitability and accuracy of selected "
            "equipment and apparatus."
        ),
        ge=1.0,
        le=5.0,
    )
    equipment_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning covering equipment selection, specifications, "
            "and operational parameters."
        ),
    )

    # Procedure Assessment (3 sub-criteria like AlchemyBench)
    procedure_completeness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the completeness and logical organization of "
            "synthesis steps."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_completeness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for procedural completeness including step "
            "order, timing, and critical parameters."
        ),
    )

    procedure_similarity_score: float = Field(
        ...,
        description=(
            "Score (1-5) for how closely the procedure matches the ground "
            "truth reference."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_similarity_reasoning: str = Field(
        ...,
        description=(
            "Detailed comparison with reference procedure highlighting "
            "matches and deviations."
        ),
    )

    procedure_feasibility_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the realistic feasibility and safety of "
            "executing the procedure in a laboratory setting."
        ),
        ge=1.0,
        le=5.0,
    )
    procedure_feasibility_reasoning: str = Field(
        ...,
        description=(
            "Detailed assessment of practical execution, safety "
            "considerations, and resource requirements."
        ),
    )

    # Characterization Assessment (2 sub-criteria like AlchemyBench)
    characterization_appropriateness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for the appropriateness of selected characterization "
            "methods for validating synthesis success."
        ),
        ge=1.0,
        le=5.0,
    )
    characterization_appropriateness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for characterization method selection and "
            "suitability for target material."
        ),
    )

    characterization_similarity_score: float = Field(
        ...,
        description=(
            "Score (1-5) for how well predicted characterization results "
            "match expected outcomes from reference."
        ),
        ge=1.0,
        le=5.0,
    )
    characterization_similarity_reasoning: str = Field(
        ...,
        description=(
            "Detailed comparison of predicted vs. expected "
            "characterization results and interpretations."
        ),
    )

    # Overall Assessment
    overall_score: float = Field(
        ...,
        description=(
            "The average of all criterion scores, representing overall "
            "synthesis recipe quality and alignment with ground truth."
        ),
        ge=1.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        ...,
        description=(
            "Comprehensive summary highlighting key strengths, weaknesses, "
            "and overall assessment of the synthesis recipe."
        ),
    )


class AlchemyBenchSynthesisEvaluation(BaseModel):
    """
    Complete evaluation following AlchemyBench methodology with enhanced
    reasoning structure.
    """

    # High-level assessment
    reasoning: str = Field(
        ...,
        description=(
            "High-level reasoning overview analyzing the synthesis recipe "
            "against the ground truth reference."
        ),
    )

    # Structured scores with detailed reasoning
    scores: AlchemyBenchSynthesisEvaluationScore = Field(
        ...,
        description=(
            "Structured evaluation scores with detailed reasoning for "
            "each criterion."
        ),
    )

    # Additional metadata for analysis
    confidence_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Judge's confidence in the evaluation based on recipe clarity and "
            "completeness."
        ),
    )

    critical_issues: list[str] = Field(
        default_factory=list,
        description=(
            "List of critical safety, feasibility, or technical issues "
            "identified."
        ),
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description=(
            "Specific recommendations for improving the synthesis recipe."
        ),
    )
