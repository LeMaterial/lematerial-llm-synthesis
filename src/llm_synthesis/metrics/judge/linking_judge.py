"""LLM judge for evaluating synthesis-to-performance linking quality.

Mirrors the architecture of ``general_synthesis_judge.py`` but is specialised
for the linking task: given the paper text, the extracted synthesis ontologies,
the extracted plot data, and the linking output, it scores the linking on four
criteria and flags specific failure modes.
"""

import json
import logging
from typing import Literal

import dspy

from llm_synthesis.metrics.judge.base import JudgeInterface
from llm_synthesis.metrics.judge.linking_evaluation_ontology import (
    LinkingEvaluation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias for the judge interface
# ---------------------------------------------------------------------------
# Input: (source_text, synthesis_json, plot_data_json, linking_output_json)
LinkingJudgeInterface = JudgeInterface[
    tuple[str, str, str, str], LinkingEvaluation
]


# ---------------------------------------------------------------------------
# DSPy judge implementation
# ---------------------------------------------------------------------------
class DspyLinkingJudge(LinkingJudgeInterface):
    """DSPy module for evaluating synthesis-to-performance linking quality.

    The judge receives:
        1. The full paper text (source of truth).
        2. The extracted synthesis ontologies (JSON list).
        3. The extracted plot data (JSON list).
        4. The linking output mapping syntheses to plot series (JSON).

    It produces a ``LinkingEvaluation`` with four criterion scores
    (1-5 in 0.5 increments), nine failure-mode flags, and supporting
    reasoning.
    """

    def __init__(
        self,
        lm: dspy.LM,
        enable_reasoning_traces: bool = False,
        confidence_threshold: float = 0.7,
        signature: type[dspy.Signature] | None = None,
    ):
        self._validate_signature(signature)
        self.signature = signature
        self.lm = lm
        self.enable_reasoning_traces = enable_reasoning_traces
        self.confidence_threshold = confidence_threshold
        super().__init__()

    # ---- public API -------------------------------------------------------

    def forward(
        self,
        input: tuple[str, str, str, str],
    ) -> LinkingEvaluation:
        """Evaluate linking output against the paper and extracted data.

        Args:
            input: Tuple of
                (source_text, synthesis_json, plot_data_json,
                 linking_output_json)

        Returns:
            A ``LinkingEvaluation`` instance.
        """
        source_text, synthesis_json, plot_data_json, linking_output_json = input

        self._validate_inputs(
            source_text, synthesis_json, plot_data_json, linking_output_json
        )

        with dspy.settings.context(
            lm=self.lm, adapter=dspy.adapters.JSONAdapter()
        ):
            prediction = dspy.Predict(self.signature)(
                source_text=source_text,
                synthesis_json=synthesis_json,
                plot_data_json=plot_data_json,
                linking_output_json=linking_output_json,
            )

            evaluation = prediction.evaluation
            evaluation = self._post_process_evaluation(evaluation)
            return evaluation

    # ---- validation -------------------------------------------------------

    def _validate_signature(self, signature: type[dspy.Signature]):
        """Check that the signature has the expected input/output fields."""
        required_inputs = {
            "source_text": str,
            "synthesis_json": str,
            "plot_data_json": str,
            "linking_output_json": str,
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
            is not LinkingEvaluation
        ):
            raise ValueError(
                "Output field 'evaluation' must be LinkingEvaluation"
            )

    def _validate_inputs(
        self,
        source_text: str,
        synthesis_json: str,
        plot_data_json: str,
        linking_output_json: str,
    ):
        """Validate that all inputs are non-trivial and well-formed JSON."""
        if not source_text or len(source_text.strip()) < 50:
            raise ValueError("Source text is too short or empty")

        for label, blob in [
            ("Synthesis JSON", synthesis_json),
            ("Plot data JSON", plot_data_json),
            ("Linking output JSON", linking_output_json),
        ]:
            if not blob or len(blob.strip()) < 10:
                raise ValueError(f"{label} is too short or empty")
            try:
                json.loads(blob)
            except json.JSONDecodeError:
                raise ValueError(f"{label} is not valid JSON")

    # ---- post-processing --------------------------------------------------

    def _post_process_evaluation(
        self, evaluation: LinkingEvaluation
    ) -> LinkingEvaluation:
        """Clamp scores, recalculate overall, and derive metadata."""
        scores = evaluation.scores
        score_fields = [
            "material_identity_score",
            "performance_data_correctness_score",
            "completeness_score",
            "format_structure_score",
        ]

        # Clamp individual scores to [1, 5] and round to nearest 0.5
        for field in score_fields:
            raw = getattr(scores, field)
            clamped = max(1.0, min(5.0, raw))
            rounded = round(clamped * 2) / 2  # snap to 0.5 increments
            setattr(scores, field, rounded)

        # Recalculate overall as the mean
        individual = [getattr(scores, f) for f in score_fields]
        scores.overall_score = round(sum(individual) / len(individual), 1)

        # Derive confidence if still at default
        if evaluation.confidence_level == "medium":
            evaluation.confidence_level = self._assess_confidence(evaluation)

        # Derive missing/spurious links from failure flags if not populated
        if not evaluation.missing_links:
            evaluation.missing_links = self._extract_missing_links(evaluation)

        if not evaluation.spurious_links:
            evaluation.spurious_links = self._extract_spurious_links(evaluation)

        if not evaluation.improvement_suggestions:
            evaluation.improvement_suggestions = self._generate_suggestions(
                evaluation
            )

        return evaluation

    def _assess_confidence(
        self, evaluation: LinkingEvaluation
    ) -> Literal["low", "medium", "high"]:
        """Heuristic confidence based on score distribution and reasoning."""
        scores = evaluation.scores
        vals = [
            scores.material_identity_score,
            scores.performance_data_correctness_score,
            scores.completeness_score,
            scores.format_structure_score,
        ]

        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)

        reasoning_length = len(evaluation.reasoning) + sum(
            len(
                getattr(
                    scores,
                    f.replace("_score", "_reasoning"),
                )
            )
            for f in [
                "material_identity_score",
                "performance_data_correctness_score",
                "completeness_score",
                "format_structure_score",
                "overall_score",
            ]
        )

        if variance < 0.5 and reasoning_length > 800 and mean > 3.5:
            return "high"
        elif variance < 1.0 and reasoning_length > 400 and mean > 2.5:
            return "medium"
        return "low"

    def _extract_missing_links(
        self, evaluation: LinkingEvaluation
    ) -> list[str]:
        missing = []
        flags = evaluation.failure_flags
        if flags.f8_false_negative:
            missing.append(
                "Algorithm missed at least one synthesis-performance pair "
                "present in the paper"
            )
        if flags.f2_one_to_many_synthesis:
            missing.append(
                "One synthesis produces multiple materials but not all "
                "were linked"
            )
        if flags.f3_many_to_one_figure:
            missing.append(
                "Multiple syntheses map to the same figure but some "
                "series were missed"
            )
        return missing

    def _extract_spurious_links(
        self, evaluation: LinkingEvaluation
    ) -> list[str]:
        spurious = []
        flags = evaluation.failure_flags
        if flags.f9_false_positive:
            spurious.append(
                "Algorithm produced a link for a material with no "
                "performance data or no synthesis in the paper"
            )
        if flags.f5_precursor_vs_product:
            spurious.append(
                "Precursor/intermediate synthesis linked to final-product "
                "performance data (or vice versa)"
            )
        if flags.f6_characterization_confusion:
            spurious.append(
                "Characterization data (XRD, TGA, BET, etc.) was confused "
                "with performance data"
            )
        return spurious

    def _generate_suggestions(self, evaluation: LinkingEvaluation) -> list[str]:
        suggestions = []
        scores = evaluation.scores
        flags = evaluation.failure_flags

        if scores.material_identity_score < 3.5:
            suggestions.append(
                "Improve material name reconciliation between text and "
                "figure legends"
            )
        if scores.performance_data_correctness_score < 3.5:
            suggestions.append(
                "Add stricter filtering to distinguish performance data "
                "from characterization data"
            )
        if scores.completeness_score < 3.5:
            suggestions.append(
                "Enhance coverage — ensure all synthesis-performance pairs "
                "in the paper are captured"
            )
        if scores.format_structure_score < 3.5:
            suggestions.append(
                "Validate output schema more strictly (axis labels, units, "
                "coordinate arrays)"
            )
        if flags.f1_name_mismatch or flags.f4_sample_code_failure:
            suggestions.append(
                "Implement fuzzy / alias-aware name matching to handle "
                "shorthand labels and naming discrepancies"
            )
        if flags.f7_dual_axis_error:
            suggestions.append(
                "Parse dual-axis plots explicitly and assign series to "
                "the correct y-axis"
            )
        return suggestions


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------
class LinkingJudgeSignature(dspy.Signature):
    """Expert-level signature for evaluating synthesis-to-performance
    linking quality."""

    source_text: str = dspy.InputField(
        description=(
            "Full text of the scientific paper (source of truth for "
            "what materials were synthesised and what performance data "
            "was reported)."
        )
    )

    synthesis_json: str = dspy.InputField(
        description=(
            "JSON list of extracted synthesis ontologies — one entry "
            "per material. Each entry follows the "
            "GeneralSynthesisOntology schema."
        )
    )

    plot_data_json: str = dspy.InputField(
        description=(
            "JSON list of extracted plot data — one entry per figure. "
            "Each entry contains series names, coordinate data, axis "
            "labels and units."
        )
    )

    linking_output_json: str = dspy.InputField(
        description=(
            "JSON output of the linking algorithm that maps each "
            "synthesis to its performance plot series. This is the "
            "artefact being evaluated."
        )
    )

    evaluation: LinkingEvaluation = dspy.OutputField(
        description=(
            """Comprehensive evaluation of synthesis-to-performance linking
quality.

EVALUATION CRITERIA (score each 1-5 in 0.5 increments):

1. Material Identity Match (material_identity_score)
   Core question: Is this the right synthesis for this data series?
   - Verify that the linked synthesis actually produced the material
     whose performance is shown in the plot series.
   - Check for name mismatches, shorthand labels, precursor vs. product
     confusion.

2. Performance Data Correctness (performance_data_correctness_score)
   Core question: Is the attached data actually relevant performance data
   from the right figure?
   - Confirm the linked plot series is genuine performance data (not
     characterization like XRD, TGA, BET, SEM).
   - Verify it comes from the correct figure and, for dual-axis plots,
     from the correct y-axis.

3. Completeness (completeness_score)
   Core question: Is anything missing or wrongly added?
   - Check that ALL valid synthesis-performance pairs in the paper are
     captured (no false negatives).
   - Check that no spurious links were introduced (no false positives).
   - Consider one-to-many and many-to-one relationships.

4. Format & Structure (format_structure_score)
   Core question: Is the final linked object well-formed and usable?
   - JSON validity, schema compliance, field presence.
   - Axis labels, units, coordinate data integrity.
   - Downstream usability (could a researcher use this as-is?).

SCORING SCALE:
  5.0  Excellent — fully correct, no issues
  4.0  Good — minor issues that do not affect correctness
  3.0  Acceptable — partially correct, some problems
  2.0  Poor — significant errors
  1.0  Failed — fundamentally wrong or entirely missing

FAILURE MODE FLAGS (set True when detected):
  F1  Name mismatch
  F2  One-to-many synthesis
  F3  Many-to-one figure
  F4  Sample code / shorthand failure
  F5  Precursor vs. product confusion
  F6  Characterization confused with performance
  F7  Dual-axis error
  F8  False negative (missed pair)
  F9  False positive (spurious link)

IMPORTANT:
- Evaluate consistency with the original text, NOT scientific accuracy.
- Do NOT penalise the linking for information that is genuinely absent
  from the paper.
- Set failure flags independently of scores — a link can score 3.0
  overall and still trigger F1 if there is a naming discrepancy that
  was partially resolved.
"""
        )
    )


def make_linking_judge_signature(
    signature_name: str = "LinkingJudgeSignature",
    instructions: str | None = None,
    source_text_description: str = ("Full paper text for linking evaluation."),
    synthesis_json_description: str = (
        "JSON list of extracted synthesis ontologies."
    ),
    plot_data_json_description: str = (
        "JSON list of extracted plot data with series and coordinates."
    ),
    linking_output_json_description: str = (
        "JSON linking output mapping syntheses to plot series."
    ),
    evaluation_description: str = (
        "Comprehensive evaluation of linking quality."
    ),
) -> type[dspy.Signature]:
    """Factory for creating a customised LinkingJudge DSPy signature.

    Follows the same pattern as
    ``make_general_synthesis_judge_signature``.
    """
    if instructions is None:
        instructions = (
            "You are an expert materials scientist evaluating how well "
            "an automated algorithm has linked extracted synthesis "
            "procedures to performance data from plots. Assess "
            "correctness, completeness, and structural quality of the "
            "linking output against the original paper text. Flag "
            "specific failure modes when detected."
        )

    signature = {
        "source_text": (
            str,
            dspy.InputField(description=source_text_description),
        ),
        "synthesis_json": (
            str,
            dspy.InputField(description=synthesis_json_description),
        ),
        "plot_data_json": (
            str,
            dspy.InputField(description=plot_data_json_description),
        ),
        "linking_output_json": (
            str,
            dspy.InputField(description=linking_output_json_description),
        ),
        "evaluation": (
            LinkingEvaluation,
            dspy.OutputField(description=evaluation_description),
        ),
    }

    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
