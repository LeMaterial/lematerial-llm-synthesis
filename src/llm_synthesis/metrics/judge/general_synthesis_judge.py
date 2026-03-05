"""General synthesis ontology judge implementation with comprehensive
evaluation capabilities for structured synthesis procedures."""

import copy
import logging
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface

log = logging.getLogger(__name__)


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

    # Overall Assessment — optional, recalculated by _post_process_evaluation
    overall_score: float = Field(
        default=0.0,
        description=(
            "The average of all criterion scores, representing overall "
            "extraction quality and ontology compliance."
        ),
        ge=0.0,
        le=5.0,
    )
    overall_reasoning: str = Field(
        default="",
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


def _get_json_schema_format() -> dict:
    """
    Return the json_schema response_format dict for GeneralSynthesisEvaluation.

    Wraps the schema under the DSPy output field name ``evaluation`` so the
    model's JSON maps directly to what DSPy's JSONAdapter expects.
    """
    inner = GeneralSynthesisEvaluation.model_json_schema()
    wrapped = {
        "type": "object",
        "properties": {"evaluation": inner},
        "required": ["evaluation"],
        "additionalProperties": True,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "GeneralSynthesisEvaluation",
            "strict": True,
            "schema": wrapped,
        },
    }


class DspyGeneralSynthesisJudge(SynthesisJudgeInterface):
    """
    Enhanced DSPy module for evaluating GeneralSynthesisOntology extraction
    quality against source synthesis text.

    Implements a two-level fallback chain for robust structured output:
      1. Strict json_schema (native for Claude/Gemini; extra_body for OpenRouter)
      2. json_object mode (valid JSON, prompt-guided schema compliance)

    Within each strategy, temperature is escalated on validation failures.
    API-level format errors (400/unsupported) skip immediately to the next
    strategy without wasting temperature retries.
    """

    def __init__(
        self,
        lm: dspy.LM,
        enable_reasoning_traces: bool = False,
        confidence_threshold: float = 0.7,
        signature: type[dspy.Signature] | None = None,
        retry_temperatures: list[float] | None = None,
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
            retry_temperatures: Temperatures to try per strategy on content
            validation failures. Defaults to [0.0, 0.3, 0.5].
        """
        self._validate_signature(signature)
        self.signature = signature
        self.enable_reasoning_traces = enable_reasoning_traces
        self.confidence_threshold = confidence_threshold
        self.retry_temperatures = retry_temperatures or [0.0, 0.3, 0.5]

        # Store a clean base LM — strip any pre-existing response_format /
        # extra_body so that _build_format_strategies() has full control.
        _clean_keys = {"response_format", "extra_body"}
        if _clean_keys & set(lm.kwargs):
            self.lm = copy.copy(lm)
            self.lm.kwargs = {
                k: v for k, v in lm.kwargs.items() if k not in _clean_keys
            }
        else:
            self.lm = lm

        super().__init__()

    # ── Public API ─────────────────────────────────────────────────────────

    def forward(
        self, input: tuple[str, str] | tuple[str, str, str]
    ) -> GeneralSynthesisEvaluation:
        """
        Evaluate extracted GeneralSynthesisOntology against source text.

        Tries each format strategy in order.  Within a strategy, retries at
        escalating temperatures on content-validation failures.  API-level
        format errors skip immediately to the next strategy.

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

        self._validate_inputs(source_text, extracted_ontology_json)

        strategies = self._build_format_strategies()
        last_exc: Exception | None = None

        for s_idx, strategy_kwargs in enumerate(strategies):
            strategy_label = (
                "json_schema" if s_idx == 0 else "json_object-fallback"
            )
            for t_idx, temp in enumerate(self.retry_temperatures):
                lm = self._lm_with_overrides(
                    {**strategy_kwargs, "temperature": temp}
                )
                try:
                    with dspy.settings.context(
                        lm=lm, adapter=dspy.adapters.JSONAdapter()
                    ):
                        prediction = dspy.Predict(self.signature)(
                            source_text=source_text,
                            extracted_ontology_json=extracted_ontology_json,
                            target_material=target_material,
                        )
                    evaluation = prediction.evaluation
                    evaluation = self._post_process_evaluation(evaluation)
                    if s_idx > 0 or t_idx > 0:
                        log.info(
                            "Judge succeeded: strategy=%s, temperature=%.1f",
                            strategy_label,
                            temp,
                        )
                    return evaluation
                except Exception as e:
                    # Recovery: model returned complete JSON but without the
                    # DSPy-expected {"evaluation": {...}} wrapper key.
                    recovered = self._try_recover_bare_json(e)
                    if recovered is not None:
                        recovered = self._post_process_evaluation(recovered)
                        log.info(
                            "Judge: recovered bare JSON response"
                            " (strategy=%s, temp=%.1f)",
                            strategy_label,
                            temp,
                        )
                        return recovered

                    last_exc = e
                    if self._is_api_format_error(e):
                        log.warning(
                            "Judge: format unsupported (%s): %r"
                            " — falling back to next strategy",
                            strategy_label,
                            e,
                        )
                        break  # skip remaining temperatures, try next strategy
                    elif t_idx < len(self.retry_temperatures) - 1:
                        log.warning(
                            "Judge: validation failure"
                            " (strategy=%s, temp=%.1f): %r"
                            " — retrying at temp=%.1f",
                            strategy_label,
                            temp,
                            e,
                            self.retry_temperatures[t_idx + 1],
                        )
                    else:
                        log.warning(
                            "Judge: all temperatures exhausted"
                            " for strategy=%s — trying next strategy",
                            strategy_label,
                        )

        raise last_exc  # type: ignore[misc]

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_format_strategies(self) -> list[dict]:
        """
        Return an ordered list of kwargs-override dicts to try, from strictest
        (json_schema) to most permissive (json_object).

        Strategy is auto-selected based on the LM's model identifier:
          - openrouter/* : extra_body workaround (LiteLLM strips response_format
            for OpenRouter, so we bypass via extra_body)
          - anthropic/*, gemini/*, others: native response_format (LiteLLM
            translates to provider format automatically)
        """
        json_schema_fmt = _get_json_schema_format()
        model: str = getattr(self.lm, "model", "")

        if "openrouter/" in model:
            # LiteLLM's OpenRouter adapter strips response_format; inject via
            # extra_body to bypass the broken detection layer.
            strict_strategy = {"extra_body": {"response_format": json_schema_fmt}}
        else:
            # Claude (anthropic/) and Gemini (gemini/) both support json_schema
            # natively; LiteLLM translates response_format automatically.
            strict_strategy = {"response_format": json_schema_fmt}

        return [
            strict_strategy,
            {"response_format": {"type": "json_object"}},
        ]

    def _lm_with_overrides(self, overrides: dict) -> dspy.LM:
        """Return a shallow copy of self.lm with kwargs overridden."""
        lm = copy.copy(self.lm)
        lm.kwargs = {**self.lm.kwargs, **overrides}
        return lm

    @staticmethod
    def _try_recover_bare_json(
        exc: Exception,
    ) -> GeneralSynthesisEvaluation | None:
        """
        Attempt to recover when the model returned a complete, valid
        GeneralSynthesisEvaluation JSON but without the DSPy-expected
        ``{"evaluation": {...}}`` wrapper key.

        DSPy's JSONAdapter embeds the raw LM response in the AdapterParseError
        message, so we can parse it directly from the exception.

        Returns the parsed evaluation on success, None otherwise.
        """
        import json
        import re

        msg = str(exc)
        # DSPy embeds the raw response as "LM Response: <json>"
        match = re.search(r"LM Response:\s*(\{)", msg, re.DOTALL)
        if not match:
            return None

        # Use raw_decode to extract the first complete JSON object starting
        # at the opening brace — stops cleanly at the end of the object.
        try:
            data, _ = json.JSONDecoder().raw_decode(msg, match.start(1))
        except json.JSONDecodeError:
            return None

        # Already wrapped — not the case we're handling
        if "evaluation" in data:
            return None

        # Try to validate as a bare GeneralSynthesisEvaluation
        try:
            return GeneralSynthesisEvaluation.model_validate(data)
        except Exception:
            return None

    @staticmethod
    def _is_api_format_error(exc: Exception) -> bool:
        """
        Return True if the exception is an API-level rejection of the requested
        response format (e.g. model doesn't support json_schema), as opposed to
        a content/validation error that warrants a temperature retry.
        """
        try:
            import litellm

            if isinstance(
                exc, (litellm.BadRequestError, litellm.UnsupportedParamsError)
            ):
                return True
        except ImportError:
            pass
        msg = str(exc).lower()
        return any(
            kw in msg
            for kw in (
                "bad request",
                "400",
                "422",
                "json_schema",
                "unsupported",
                "invalid request",
                "response_format",
            )
        )

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

CORE PRINCIPLE — ABSENCE IS NOT AN ERROR:
The extraction system must never hallucinate. When a field is null or
empty, ask: "Is this stated in the source text?" If no → CORRECT.
Only penalize when the source clearly states something the extractor
missed or got wrong.

Examples of CORRECT behavior that must NOT be penalized:
- Null reagent amounts/units → paper does not state quantities
- Null equipment vendor/model → paper does not name a vendor
- Null step duration, atmosphere, or pressure → not specified in paper
- Empty steps/equipment/materials → no synthesis for this material
  (e.g. commercial reference: "20% Pt/C was used as received")
- Generic or absent precursor name → paper does not identify it

EVALUATION CRITERIA:
1. Structural Completeness (1-5): Coverage of all synthesis components
EXPLICITLY PRESENT in the source text. Do not penalize for null fields
corresponding to information absent from the paper.
2. Material Extraction (1-5): Accuracy of materials, quantities, and units
extracted from the paper. Null amounts/purities are correct when the paper
does not state them — do not penalize.
3. Process Steps (1-5): Correct sequencing and classification of synthesis
actions present in the source. Empty steps are correct for materials with
no described synthesis procedure.
4. Equipment Extraction (1-5): Identification of equipment EXPLICITLY NAMED
in the text. Do not penalize for absent vendor info or for not inferring
equipment from implied processes (e.g. "annealed" does not require naming
a furnace if none is mentioned).
5. Conditions Extraction (1-5): Accurate representation of synthesis conditions
STATED in the source. Null duration, atmosphere, or pressure are correct when
the paper does not provide these values.
6. Semantic Accuracy (1-5): Faithful preservation of the scientific meaning
from the original text
7. Format Compliance (1-5): Adherence to the specified schema and data types

EVALUATION APPROACH:
- Compare extracted ontology against the source text carefully and
systematically
- Assess accuracy, completeness (relative to the paper), and semantic fidelity
- Identify and explain any extraction errors or misinterpretations
- Verify schema conformity and data type correctness
- Provide clear reasoning for each score
- Suggest actionable improvements if applicable

SCORING GUIDELINES:
- 5.0: Excellent - Accurate, complete (w.r.t. source), and semantically faithful
- 4.0-4.5: Good - Minor omissions or minor semantic shifts
- 3.0-3.5: Adequate - Noticeable issues, but mostly acceptable
- 2.0-2.5: Poor - Significant inaccuracies or misunderstandings
- 1.0-1.5: Very Poor - Major errors or misrepresentations

CRITICAL REQUIREMENT: You MUST populate EVERY field without exception:
- reasoning, confidence_level
- scores.structural_completeness_score AND
  scores.structural_completeness_reasoning
- scores.material_extraction_score AND scores.material_extraction_reasoning
- scores.process_steps_score AND scores.process_steps_reasoning
- scores.equipment_extraction_score AND scores.equipment_extraction_reasoning
- scores.conditions_extraction_score AND scores.conditions_extraction_reasoning
- scores.semantic_accuracy_score AND scores.semantic_accuracy_reasoning
- scores.format_compliance_score AND scores.format_compliance_reasoning
- scores.overall_reasoning
Do NOT omit any field. An incomplete response is invalid.

Focus on scientific accuracy, structural integrity, and source faithfulness.
"""
        )
    )


def make_judge_extra_body() -> dict:
    """
    Build the extra_body dict for OpenRouter models that need explicit
    structured-output enforcement via json_schema response_format.

    Note: DspyGeneralSynthesisJudge now handles this automatically based on
    the model name. This function is kept for external use / testing.

    Returns:
        extra_body dict containing the json_schema response_format.
    """
    return {"response_format": _get_json_schema_format()}


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
        "Comprehensive evaluation of ontology extraction quality. "
        "CRITICAL: populate ALL fields — reasoning, confidence_level, "
        "all seven *_score and *_reasoning pairs inside scores, and "
        "scores.overall_reasoning. Omitting any field is invalid."
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
