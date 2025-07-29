"""General synthesis ontology judge implementation with comprehensive
evaluation capabilities for structured synthesis procedures."""

from typing import Literal

import dspy
from pydantic import BaseModel, Field

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface


class GeneralSynthesisEvaluationScore(BaseModel):
    """
    Evaluation scores for GeneralSynthesisOntology extraction quality.
    Scores are on a scale of 1.0 (poor) to 5.0 (excellent) with 0.5
    increments.
    """

    # Structural Completeness Assessment
    structural_completeness_score: float = Field(
        ...,
        description=(
            "Score (1-5) for how completely the structured ontology "
            "captures all synthesis information from the source text."
        ),
        ge=1.0,
        le=5.0,
    )
    structural_completeness_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for structural completeness including "
            "coverage of materials, steps, equipment, and conditions."
        ),
    )

    # Material Extraction Assessment
    material_extraction_score: float = Field(
        ...,
        description=(
            "Score (1-5) for accuracy and completeness of material "
            "extraction including names, amounts, units, and purities."
        ),
        ge=1.0,
        le=5.0,
    )
    material_extraction_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for material extraction quality including "
            "accuracy of quantities, units, and chemical names."
        ),
    )

    # Process Steps Assessment
    process_steps_score: float = Field(
        ...,
        description=(
            "Score (1-5) for accuracy and organization of process steps "
            "including correct sequencing and action classification."
        ),
        ge=1.0,
        le=5.0,
    )
    process_steps_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for process steps quality including "
            "logical flow, completeness, and action accuracy."
        ),
    )

    # Equipment Extraction Assessment
    equipment_extraction_score: float = Field(
        ...,
        description=(
            "Score (1-5) for completeness and accuracy of equipment "
            "extraction including names, vendors, and settings."
        ),
        ge=1.0,
        le=5.0,
    )
    equipment_extraction_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for equipment extraction including "
            "identification accuracy and technical specifications."
        ),
    )

    # Conditions Extraction Assessment
    conditions_extraction_score: float = Field(
        ...,
        description=(
            "Score (1-5) for accuracy of synthesis conditions extraction "
            "including temperature, pressure, duration, and atmosphere."
        ),
        ge=1.0,
        le=5.0,
    )
    conditions_extraction_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for conditions extraction including "
            "numerical accuracy and unit consistency."
        ),
    )

    # Semantic Accuracy Assessment
    semantic_accuracy_score: float = Field(
        ...,
        description=(
            "Score (1-5) for semantic accuracy and preservation of "
            "scientific meaning in the structured format."
        ),
        ge=1.0,
        le=5.0,
    )
    semantic_accuracy_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for semantic accuracy including "
            "preservation of scientific context and meaning."
        ),
    )

    # Format Compliance Assessment
    format_compliance_score: float = Field(
        ...,
        description=(
            "Score (1-5) for adherence to the GeneralSynthesisOntology "
            "schema and data type requirements."
        ),
        ge=1.0,
        le=5.0,
    )
    format_compliance_reasoning: str = Field(
        ...,
        description=(
            "Detailed reasoning for format compliance including "
            "schema adherence and data type correctness."
        ),
    )

    # Overall Assessment
    overall_score: float = Field(
        ...,
        description=(
            "The average of all criterion scores, representing overall "
            "extraction quality and ontology compliance."
        ),
        ge=1.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        ...,
        description=(
            "Comprehensive summary highlighting key strengths, weaknesses, "
            "and overall assessment of the ontology extraction."
        ),
    )


class GeneralSynthesisEvaluation(BaseModel):
    """
    Complete evaluation of GeneralSynthesisOntology extraction quality.
    """

    # High-level assessment
    reasoning: str = Field(
        ...,
        description=(
            "High-level reasoning overview analyzing the extracted ontology "
            "against the source synthesis text."
        ),
    )

    # Structured scores with detailed reasoning
    scores: GeneralSynthesisEvaluationScore = Field(
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
            "Judge's confidence in the evaluation based on extraction "
            "clarity and completeness."
        ),
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "List of important synthesis information that was not "
            "captured in the structured format."
        ),
    )

    extraction_errors: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific errors or inaccuracies in the extraction."
        ),
    )

    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Specific suggestions for improving the ontology extraction."
        ),
    )


class DspyGeneralSynthesisJudge(SynthesisJudgeInterface):
    """
    Enhanced DSPy module for evaluating GeneralSynthesisOntology extraction
    quality against source synthesis text.
    """

    def __init__(
        self,
        lm: dspy.LM,
        enable_reasoning_traces: bool = False,
        confidence_threshold: float = 0.7,
        signature: type[dspy.Signature] | None = None,
    ):
        """
        Initialize the unified synthesis judge.

        Args:
            signature: DSPy signature for evaluation
            lm: Language model for evaluation
            enable_reasoning_traces: Whether to include detailed reasoning
            traces
            confidence_threshold: Minimum confidence threshold for reliable
            evaluations
        """
        self._validate_signature(signature)
        self.signature = signature
        self.lm = lm
        self.enable_reasoning_traces = enable_reasoning_traces
        self.confidence_threshold = confidence_threshold
        super().__init__()

    def forward(
        self, input: tuple[str, str] | tuple[str, str, str]
    ) -> GeneralSynthesisEvaluation:
        """
        Evaluate extracted GeneralSynthesisOntology against source text.

        Args:
            input: Tuple of (source_text, extracted_ontology_json) or
                   (source_text, extracted_ontology_json, target_material)

        Returns:
            Comprehensive evaluation of the ontology extraction
        """
        if len(input) == 2:
            source_text, extracted_ontology_json = input
            target_material = self._extract_target_from_json(
                extracted_ontology_json
            )
        else:
            source_text, extracted_ontology_json, target_material = input

        # Validate inputs
        self._validate_inputs(source_text, extracted_ontology_json)

        # Perform evaluation
        with dspy.settings.context(
            lm=self.lm, adapter=dspy.adapters.JSONAdapter()
        ):
            prediction = dspy.Predict(self.signature)(
                source_text=source_text,
                extracted_ontology_json=extracted_ontology_json,
                target_material=target_material,
            )

            evaluation = prediction.evaluation

            # Post-process evaluation
            evaluation = self._post_process_evaluation(evaluation)

            return evaluation

    def _validate_signature(self, signature: type[dspy.Signature]):
        """Validate that the signature contains all required fields."""
        required_inputs = {
            "source_text": str,
            "extracted_ontology_json": str,
            "target_material": str,
        }

        for field_name, field_type in required_inputs.items():
            if field_name not in signature.input_fields:
                raise ValueError(
                    f"Required input field '{field_name}' missing from "
                    f"signature"
                )
            if signature.input_fields[field_name].annotation is not field_type:
                raise ValueError(
                    f"Input field '{field_name}' must be {field_type}"
                )

        if "evaluation" not in signature.output_fields:
            raise ValueError(
                "Required output field 'evaluation' missing from signature"
            )
        if (
            signature.output_fields["evaluation"].annotation
            is not GeneralSynthesisEvaluation
        ):
            raise ValueError(
                "Output field 'evaluation' must be GeneralSynthesisEvaluation"
            )

    def _extract_target_from_json(self, ontology_json: str) -> str:
        """Extract target material from the ontology JSON."""
        try:
            import json

            data = json.loads(ontology_json)
            return data.get("target_compound", "Unknown target material")
        except Exception:
            return "Unknown target material"

    def _validate_inputs(self, source_text: str, extracted_ontology_json: str):
        """Validate input quality and completeness."""
        if not source_text or len(source_text.strip()) < 50:
            raise ValueError("Source text is too short or empty")

        if (
            not extracted_ontology_json
            or len(extracted_ontology_json.strip()) < 20
        ):
            raise ValueError("Extracted ontology JSON is too short or empty")

        # Validate JSON format
        try:
            import json

            json.loads(extracted_ontology_json)
        except json.JSONDecodeError:
            raise ValueError("Extracted ontology is not valid JSON")

    def _post_process_evaluation(
        self, evaluation: GeneralSynthesisEvaluation
    ) -> GeneralSynthesisEvaluation:
        """Post-process evaluation for consistency and derived metrics."""
        scores = evaluation.scores

        # Validate and clamp scores
        score_fields = [
            "structural_completeness_score",
            "material_extraction_score",
            "process_steps_score",
            "equipment_extraction_score",
            "conditions_extraction_score",
            "semantic_accuracy_score",
            "format_compliance_score",
        ]

        for field in score_fields:
            score = getattr(scores, field)
            if not (1.0 <= score <= 5.0):
                clamped_score = max(1.0, min(5.0, score))
                setattr(scores, field, clamped_score)

        # Recalculate overall score
        individual_scores = [getattr(scores, field) for field in score_fields]
        calculated_overall = sum(individual_scores) / len(individual_scores)
        scores.overall_score = round(calculated_overall, 1)

        # Assess confidence if not set
        if evaluation.confidence_level == "medium":
            evaluation.confidence_level = self._assess_confidence(evaluation)

        # Extract issues and suggestions if not present
        if not evaluation.missing_information:
            evaluation.missing_information = self._extract_missing_info(
                evaluation
            )

        if not evaluation.extraction_errors:
            evaluation.extraction_errors = self._extract_errors(evaluation)

        if not evaluation.improvement_suggestions:
            evaluation.improvement_suggestions = self._generate_suggestions(
                evaluation
            )

        return evaluation

    def _assess_confidence(self, evaluation: GeneralSynthesisEvaluation) -> str:
        """Assess confidence level based on scores and reasoning quality."""
        scores = evaluation.scores
        score_values = [
            scores.structural_completeness_score,
            scores.material_extraction_score,
            scores.process_steps_score,
            scores.equipment_extraction_score,
            scores.conditions_extraction_score,
            scores.semantic_accuracy_score,
            scores.format_compliance_score,
        ]

        mean_score = sum(score_values) / len(score_values)
        variance = sum(
            (score - mean_score) ** 2 for score in score_values
        ) / len(score_values)

        reasoning_length = len(evaluation.reasoning) + sum(
            len(getattr(scores, f"{field}_reasoning"))
            for field in [
                "structural_completeness",
                "material_extraction",
                "process_steps",
                "equipment_extraction",
                "conditions_extraction",
                "semantic_accuracy",
                "format_compliance",
                "overall",
            ]
        )

        if variance < 0.5 and reasoning_length > 1000 and mean_score > 3.5:
            return "high"
        elif variance < 1.0 and reasoning_length > 500 and mean_score > 2.5:
            return "medium"
        else:
            return "low"

    def _extract_missing_info(
        self, evaluation: GeneralSynthesisEvaluation
    ) -> list[str]:
        """Extract missing information from low scores."""
        missing = []
        scores = evaluation.scores

        if scores.material_extraction_score < 3.0:
            missing.append("Material quantities, units, or purities")

        if scores.process_steps_score < 3.0:
            missing.append("Process step details or sequencing")

        if scores.equipment_extraction_score < 3.0:
            missing.append("Equipment specifications or settings")

        if scores.conditions_extraction_score < 3.0:
            missing.append(
                "Synthesis conditions (temperature, pressure, duration)"
            )

        return missing

    def _extract_errors(
        self, evaluation: GeneralSynthesisEvaluation
    ) -> list[str]:
        """Extract errors from reasoning text."""
        errors = []
        scores = evaluation.scores

        if scores.semantic_accuracy_score < 2.5:
            errors.append("Semantic meaning not preserved in structured format")

        if scores.format_compliance_score < 2.5:
            errors.append("Schema compliance issues or data type errors")

        return errors

    def _generate_suggestions(
        self, evaluation: GeneralSynthesisEvaluation
    ) -> list[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []
        scores = evaluation.scores

        if scores.structural_completeness_score < 3.5:
            suggestions.append("Improve coverage of all synthesis components")

        if scores.material_extraction_score < 3.5:
            suggestions.append(
                "Enhance material parsing for quantities and units"
            )

        if scores.process_steps_score < 3.5:
            suggestions.append("Better organize and sequence process steps")

        if scores.format_compliance_score < 3.5:
            suggestions.append("Ensure strict adherence to ontology schema")

        return suggestions


class GeneralSynthesisJudgeSignature(dspy.Signature):
    """
    Expert-level signature for evaluating GeneralSynthesisOntology extraction
    quality against source synthesis text.
    """

    source_text: str = dspy.InputField(
        description=(
            "Original synthesis text or paragraph from which the structured "
            "ontology should be extracted."
        )
    )

    extracted_ontology_json: str = dspy.InputField(
        description=(
            "JSON representation of the extracted GeneralSynthesisOntology "
            "with all structured components."
        )
    )

    target_material: str = dspy.InputField(
        description=(
            "Target material or compound being synthesized, used for "
            "context in evaluation."
        )
    )

    evaluation: GeneralSynthesisEvaluation = dspy.OutputField(
        description=(
            """Comprehensive evaluation of GeneralSynthesisOntology extraction
quality.

EVALUATION CRITERIA:
1. Structural Completeness (1-5): Coverage of all synthesis components
2. Material Extraction (1-5): Accuracy of materials, quantities, units
3. Process Steps (1-5): Correct sequencing and action classification
4. Equipment Extraction (1-5): Complete equipment identification
5. Conditions Extraction (1-5): Accurate synthesis conditions
6. Semantic Accuracy (1-5): Preservation of scientific meaning
7. Format Compliance (1-5): Schema adherence and data types

EVALUATION APPROACH:
- Compare extracted ontology against source text systematically
- Assess completeness, accuracy, and semantic preservation
- Identify missing information and extraction errors
- Evaluate schema compliance and data type correctness
- Provide detailed reasoning for each criterion
- Generate actionable improvement suggestions

SCORING GUIDELINES:
- 5.0: Excellent - Complete, accurate, semantically preserved
- 4.0-4.5: Good - Minor gaps but high quality extraction
- 3.0-3.5: Adequate - Some issues but generally acceptable
- 2.0-2.5: Poor - Significant gaps or inaccuracies
- 1.0-1.5: Very Poor - Major missing components or errors

Focus on scientific accuracy, completeness, and structural integrity."""
        )
    )


def make_general_synthesis_judge_signature(
    signature_name: str = "GeneralSynthesisJudgeSignature",
    instructions: str | None = None,
    source_text_description: str = (
        "Original synthesis text for ontology extraction evaluation."
    ),
    extracted_ontology_description: str = (
        "JSON representation of extracted GeneralSynthesisOntology."
    ),
    target_material_description: str = (
        "Target material for synthesis context."
    ),
    evaluation_description: str = (
        "Comprehensive evaluation of ontology extraction quality."
    ),
) -> type[dspy.Signature]:
    """
    Create a DSPy signature for GeneralSynthesisOntology evaluation.

    Args:
        signature_name: Name of the signature class
        instructions: Custom instructions for the evaluation
        source_text_description: Description for source text input
        extracted_ontology_description: Description for ontology JSON input
        target_material_description: Description for target material input
        evaluation_description: Description for evaluation output

    Returns:
        DSPy signature class for ontology evaluation
    """
    if instructions is None:
        instructions = (
            "You are an expert in materials science and data extraction. "
            "Evaluate how well the GeneralSynthesisOntology extraction "
            "captures all synthesis information from the source text. "
            "Assess completeness, accuracy, and semantic preservation "
            "across all ontology components."
        )

    signature = {
        "source_text": (
            str,
            dspy.InputField(description=source_text_description),
        ),
        "extracted_ontology_json": (
            str,
            dspy.InputField(description=extracted_ontology_description),
        ),
        "target_material": (
            str,
            dspy.InputField(description=target_material_description),
        ),
        "evaluation": (
            GeneralSynthesisEvaluation,
            dspy.OutputField(description=evaluation_description),
        ),
    }

    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
