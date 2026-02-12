"""Evaluation ontology for synthesis-to-performance linking quality.

Defines scoring criteria and failure mode flags for assessing whether
the LLM linking procedure correctly associates extracted synthesis
procedures with performance data from plots.
"""

from typing import Literal

from pydantic import BaseModel, Field


class LinkingFailureFlags(BaseModel):
    """Failure mode flags for diagnosing linking errors.

    Each flag is True when the corresponding failure mode is detected.
    Multiple flags can be active simultaneously.
    """

    f1_name_mismatch: bool = Field(
        default=False,
        description=(
            "F1 — Text and figure use different names for the same "
            "material and the algorithm failed to reconcile them."
        ),
    )
    f2_one_to_many_synthesis: bool = Field(
        default=False,
        description=(
            "F2 — One synthesis produces multiple materials but only "
            "one was linked."
        ),
    )
    f3_many_to_one_figure: bool = Field(
        default=False,
        description=(
            "F3 — Multiple syntheses should link to different series "
            "in the same figure; algorithm missed some or conflated them."
        ),
    )
    f4_sample_code_failure: bool = Field(
        default=False,
        description=(
            "F4 — Paper uses shorthand labels (e.g. 'S1', 'NCA-3') "
            "that the algorithm could not resolve."
        ),
    )
    f5_precursor_vs_product: bool = Field(
        default=False,
        description=(
            "F5 — Synthesis of a precursor or intermediate was linked "
            "to final-product performance data (or vice versa)."
        ),
    )
    f6_characterization_confusion: bool = Field(
        default=False,
        description=(
            "F6 — Performance data confused with characterization data "
            "(XRD, TGA, BET, SEM, etc.)."
        ),
    )
    f7_dual_axis_error: bool = Field(
        default=False,
        description=(
            "F7 — Data series attributed to the wrong y-axis on a "
            "dual-axis plot."
        ),
    )
    f8_false_negative: bool = Field(
        default=False,
        description=(
            "F8 — Paper clearly has a synthesis + performance pair "
            "that the algorithm missed entirely."
        ),
    )
    f9_false_positive: bool = Field(
        default=False,
        description=(
            "F9 — Algorithm produced a link for a material with no "
            "performance data, or no synthesis in the paper."
        ),
    )

    def active_flags(self) -> list[str]:
        """Return a list of flag codes that are currently active."""
        flag_map = {
            "f1_name_mismatch": "F1",
            "f2_one_to_many_synthesis": "F2",
            "f3_many_to_one_figure": "F3",
            "f4_sample_code_failure": "F4",
            "f5_precursor_vs_product": "F5",
            "f6_characterization_confusion": "F6",
            "f7_dual_axis_error": "F7",
            "f8_false_negative": "F8",
            "f9_false_positive": "F9",
        }
        return [
            code
            for field_name, code in flag_map.items()
            if getattr(self, field_name)
        ]


class LinkingEvaluationScore(BaseModel):
    """Scoring rubric for synthesis-to-performance linking quality.

    Each criterion is scored 1.0-5.0 in 0.5 increments.

    Scale:
        5.0  Excellent — fully correct, no issues
        4.0  Good — minor issues that do not affect correctness
        3.0  Acceptable — partially correct, some problems
        2.0  Poor — significant errors
        1.0  Failed — fundamentally wrong or entirely missing
    """

    # Criterion 1 — Material Identity Match
    material_identity_score: float = Field(
        ...,
        description=(
            "Score (1-5): Is this the right synthesis for this data series? "
            "Checks whether each linked synthesis actually produced the "
            "material whose performance is shown."
        ),
        ge=1.0,
        le=5.0,
    )
    material_identity_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for material identity score, including "
            "which material names were checked and how they correspond."
        ),
    )

    # Criterion 2 — Performance Data Correctness
    performance_data_correctness_score: float = Field(
        ...,
        description=(
            "Score (1-5): Is the attached data actually relevant "
            "performance data from the correct figure? Checks that the "
            "linked plot series is genuine performance data (not "
            "characterization), from the right figure, and from the "
            "right data series / y-axis."
        ),
        ge=1.0,
        le=5.0,
    )
    performance_data_correctness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for performance data correctness, "
            "including figure reference verification and data type check."
        ),
    )

    # Criterion 3 — Completeness
    completeness_score: float = Field(
        ...,
        description=(
            "Score (1-5): Is anything missing or wrongly added? "
            "Assesses whether all valid synthesis-performance pairs in "
            "the paper were captured and whether any spurious links "
            "were introduced."
        ),
        ge=1.0,
        le=5.0,
    )
    completeness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for completeness, listing any missed "
            "links (false negatives) or spurious links (false positives)."
        ),
    )

    # Criterion 4 — Format & Structure
    format_structure_score: float = Field(
        ...,
        description=(
            "Score (1-5): Is the final linked object well-formed and "
            "usable? Checks JSON structure, field presence, axis labels, "
            "units, coordinate data integrity, and downstream usability."
        ),
        ge=1.0,
        le=5.0,
    )
    format_structure_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for format and structure quality, "
            "including schema compliance and data integrity checks."
        ),
    )

    # Overall
    overall_score: float = Field(
        ...,
        description=(
            "Average of the four criterion scores, representing overall "
            "linking quality."
        ),
        ge=1.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        ...,
        description=(
            "Comprehensive summary of strengths, weaknesses, and overall "
            "assessment of the linking quality."
        ),
    )


class LinkingEvaluation(BaseModel):
    """Complete evaluation of synthesis-to-performance linking quality."""

    # High-level assessment
    reasoning: str = Field(
        ...,
        description=(
            "High-level reasoning overview analysing the linking output "
            "against the original paper text and plot data."
        ),
    )

    # Structured scores
    scores: LinkingEvaluationScore = Field(
        ...,
        description=(
            "Structured evaluation scores with detailed reasoning for "
            "each of the four linking criteria."
        ),
    )

    # Failure mode flags
    failure_flags: LinkingFailureFlags = Field(
        default_factory=LinkingFailureFlags,
        description=(
            "Failure mode flags indicating specific categories of "
            "linking errors detected during evaluation."
        ),
    )

    # Metadata
    confidence_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Judge's confidence in this evaluation based on how clearly "
            "the paper presents synthesis and performance information."
        ),
    )

    missing_links: list[str] = Field(
        default_factory=list,
        description=(
            "List of synthesis-performance pairs present in the paper "
            "but absent from the linking output (false negatives)."
        ),
    )

    spurious_links: list[str] = Field(
        default_factory=list,
        description=(
            "List of links produced by the algorithm that are incorrect "
            "or not supported by the paper (false positives)."
        ),
    )

    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, actionable suggestions for improving the linking "
            "algorithm."
        ),
    )
