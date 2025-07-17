"""AlchemyBench-style synthesis judge implementation with enhanced evaluation
capabilities."""

import dspy

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface
from llm_synthesis.models.ontologies.alchemybench import (
    AlchemyBenchSynthesisEvaluation,
)


class DspyAlchemyBenchSynthesisJudge(SynthesisJudgeInterface):
    """
    Enhanced DSPy module implementing AlchemyBench methodology for evaluating
    LLM-extracted synthesis recipes against reference procedures.

    Key improvements over the basic implementation:
    - 7-criteria evaluation framework from AlchemyBench
    - Enhanced reasoning structure with confidence assessment
    - Critical issue identification and recommendations
    - Context-aware evaluation considering synthesis domain
    - Improved prompt engineering for materials science expertise
    """

    def __init__(
        self,
        signature: type[dspy.Signature],
        lm: dspy.LM,
        enable_reasoning_traces: bool = False,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the enhanced AlchemyBench judge.

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
        self, input: tuple[str, str, str] | tuple[str, str, str, str]
    ) -> AlchemyBenchSynthesisEvaluation:
        """
        Enhanced forward method with context awareness and validation.

        Args:
            input: Tuple of (target_material, extracted_recipe,
                   reference_procedure) or (target_material, extracted_recipe,
                   reference_procedure, synthesis_context)

        Returns:
            Comprehensive AlchemyBench-style evaluation
        """
        if len(input) == 3:
            # Backward compatibility with existing 3-tuple input
            target_material, extracted_recipe, reference_procedure = input
            synthesis_context = self._infer_synthesis_context(
                target_material, extracted_recipe
            )
        else:
            (
                target_material,
                extracted_recipe,
                reference_procedure,
                synthesis_context,
            ) = input

        # Validate inputs
        self._validate_inputs(
            target_material, extracted_recipe, reference_procedure
        )

        # Perform evaluation with enhanced context
        with dspy.settings.context(lm=self.lm):
            prediction = dspy.ChainOfThought(self.signature)(
                target_material=target_material,
                extracted_recipe=extracted_recipe,
                reference_procedure=reference_procedure,
                synthesis_context=synthesis_context,
            )

            evaluation = prediction.evaluation

            # Post-process evaluation
            evaluation = self._post_process_evaluation(evaluation)

            return evaluation

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains all required input and output
        fields.

        Args:
            signature: The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        required_inputs = {
            "target_material": str,
            "extracted_recipe": str,
            "reference_procedure": str,
            "synthesis_context": str,
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
            is not AlchemyBenchSynthesisEvaluation
        ):
            raise ValueError(
                "Output field 'evaluation' must be "
                "AlchemyBenchSynthesisEvaluation"
            )

    def _infer_synthesis_context(
        self, target_material: str, extracted_recipe: str
    ) -> str:
        """
        Infer synthesis context from target material and recipe for backward
        compatibility.
        """
        context_elements = []

        # Material class inference
        target_lower = target_material.lower()
        recipe_lower = extracted_recipe.lower()

        if any(
            term in target_lower for term in ["oxide", "ceramic", "perovskite"]
        ):
            context_elements.append("Material class: Ceramic/Oxide")
        elif any(term in target_lower for term in ["polymer", "organic"]):
            context_elements.append("Material class: Polymer/Organic")
        elif any(
            term in target_lower for term in ["metal", "alloy", "nanoparticle"]
        ):
            context_elements.append("Material class: Metallic/Nanoparticle")
        elif any(
            term in target_lower for term in ["semiconductor", "quantum"]
        ):
            context_elements.append("Material class: Semiconductor")

        # Application domain inference
        if any(
            term in target_lower
            for term in ["battery", "electrode", "lithium", "energy storage"]
        ):
            context_elements.append("Application domain: Energy storage")
        elif any(term in target_lower for term in ["catalyst", "catalytic"]):
            context_elements.append("Application domain: Catalysis")
        elif any(
            term in target_lower
            for term in ["sensor", "electronic", "transistor"]
        ):
            context_elements.append("Application domain: Electronics")
        elif any(term in target_lower for term in ["photovoltaic", "solar"]):
            context_elements.append("Application domain: Photovoltaics")

        # Synthesis method inference
        if any(
            term in recipe_lower for term in ["hydrothermal", "solvothermal"]
        ):
            context_elements.append(
                "Synthesis method: Hydrothermal/Solvothermal"
            )
        elif any(term in recipe_lower for term in ["sol-gel", "solution"]):
            context_elements.append("Synthesis method: Solution-based")
        elif any(
            term in recipe_lower
            for term in ["calcination", "sintering", "solid-state"]
        ):
            context_elements.append("Synthesis method: Solid-state")
        elif any(
            term in recipe_lower
            for term in ["cvd", "chemical vapor deposition"]
        ):
            context_elements.append(
                "Synthesis method: Chemical vapor deposition"
            )
        elif any(
            term in recipe_lower
            for term in ["electrochemical", "electrodeposition"]
        ):
            context_elements.append("Synthesis method: Electrochemical")

        return (
            "; ".join(context_elements)
            if context_elements
            else "General materials synthesis"
        )

    def _validate_inputs(
        self,
        target_material: str,
        extracted_recipe: str,
        reference_procedure: str,
    ):
        """
        Validate input quality and completeness.
        """
        if not target_material or len(target_material.strip()) < 10:
            raise ValueError(
                "Target material description is too short or empty"
            )

        if not extracted_recipe or len(extracted_recipe.strip()) < 50:
            raise ValueError("Extracted recipe is too short or empty")

        if not reference_procedure or len(reference_procedure.strip()) < 50:
            raise ValueError("Reference procedure is too short or empty")

    def _post_process_evaluation(
        self, evaluation: AlchemyBenchSynthesisEvaluation
    ) -> AlchemyBenchSynthesisEvaluation:
        """
        Post-process evaluation to ensure consistency and add derived metrics.
        """
        scores = evaluation.scores

        # Validate score ranges and clamp if necessary
        score_fields = [
            "materials_appropriateness_score",
            "equipment_appropriateness_score",
            "procedure_completeness_score",
            "procedure_similarity_score",
            "procedure_feasibility_score",
            "characterization_appropriateness_score",
            "characterization_similarity_score",
        ]

        for field in score_fields:
            score = getattr(scores, field)
            if not (1.0 <= score <= 5.0):
                # Clamp to valid range
                clamped_score = max(1.0, min(5.0, score))
                setattr(scores, field, clamped_score)

        # Recalculate overall score as average
        individual_scores = [getattr(scores, field) for field in score_fields]
        calculated_overall = sum(individual_scores) / len(individual_scores)
        scores.overall_score = round(calculated_overall, 1)

        # Add confidence assessment if not present or default
        if evaluation.confidence_level == "medium":  # Default value
            evaluation.confidence_level = self._assess_confidence(evaluation)

        # Extract critical issues if not present
        if not evaluation.critical_issues:
            evaluation.critical_issues = self._extract_critical_issues(
                evaluation
            )

        # Generate recommendations if not present
        if not evaluation.recommendations:
            evaluation.recommendations = self._generate_recommendations(
                evaluation
            )

        return evaluation

    def _assess_confidence(
        self, evaluation: AlchemyBenchSynthesisEvaluation
    ) -> str:
        """
        Assess confidence level based on score variance and reasoning quality.
        """
        scores = evaluation.scores
        score_values = [
            scores.materials_appropriateness_score,
            scores.equipment_appropriateness_score,
            scores.procedure_completeness_score,
            scores.procedure_similarity_score,
            scores.procedure_feasibility_score,
            scores.characterization_appropriateness_score,
            scores.characterization_similarity_score,
        ]

        # Calculate score variance
        mean_score = sum(score_values) / len(score_values)
        variance = sum(
            (score - mean_score) ** 2 for score in score_values
        ) / len(score_values)

        # Assess reasoning quality (simple heuristic)
        reasoning_length = len(evaluation.reasoning) + sum(
            len(getattr(scores, f"{field}_reasoning"))
            for field in [
                "materials_appropriateness",
                "equipment_appropriateness",
                "procedure_completeness",
                "procedure_similarity",
                "procedure_feasibility",
                "characterization_appropriateness",
                "characterization_similarity",
                "overall",
            ]
        )

        # Determine confidence level
        if variance < 0.5 and reasoning_length > 1000 and mean_score > 3.0:
            return "high"
        elif variance < 1.0 and reasoning_length > 500:
            return "medium"
        else:
            return "low"

    def _extract_critical_issues(
        self, evaluation: AlchemyBenchSynthesisEvaluation
    ) -> list[str]:
        """
        Extract critical issues from low scores and reasoning.
        """
        issues = []
        scores = evaluation.scores

        # Check for critically low scores
        if scores.materials_appropriateness_score < 2.5:
            issues.append("Critical material selection or quantity issues")

        if scores.equipment_appropriateness_score < 2.5:
            issues.append(
                "Inappropriate or inadequate equipment specification"
            )

        if scores.procedure_feasibility_score < 2.5:
            issues.append(
                "Procedure may not be safely or practically executable"
            )

        if scores.procedure_completeness_score < 2.5:
            issues.append("Incomplete or poorly organized procedure")

        # Check for safety keywords in reasoning
        safety_keywords = [
            "unsafe",
            "dangerous",
            "hazardous",
            "toxic",
            "explosive",
            "risk",
        ]
        all_reasoning = (
            f"{evaluation.reasoning} {scores.overall_reasoning}".lower()
        )

        if any(keyword in all_reasoning for keyword in safety_keywords):
            issues.append("Potential safety concerns identified")

        return issues

    def _generate_recommendations(
        self, evaluation: AlchemyBenchSynthesisEvaluation
    ) -> list[str]:
        """
        Generate recommendations based on evaluation scores and reasoning.
        """
        recommendations = []
        scores = evaluation.scores

        # Materials recommendations
        if scores.materials_appropriateness_score < 3.5:
            recommendations.append(
                "Review material selection, quantities, and purities against "
                "established protocols"
            )

        # Equipment recommendations
        if scores.equipment_appropriateness_score < 3.5:
            recommendations.append(
                "Verify equipment specifications and operating parameters"
            )

        # Procedure recommendations
        if scores.procedure_completeness_score < 3.5:
            recommendations.append(
                "Provide more detailed step-by-step procedures with specific "
                "parameters"
            )

        if scores.procedure_feasibility_score < 3.5:
            recommendations.append(
                "Consider practical constraints and safety requirements for "
                "laboratory execution"
            )

        # Characterization recommendations
        if scores.characterization_appropriateness_score < 3.5:
            recommendations.append(
                "Select more appropriate characterization methods for the "
                "target material"
            )

        # Overall recommendations
        if scores.overall_score < 3.0:
            recommendations.append(
                "Consider consulting established synthesis protocols and "
                "expert guidance"
            )

        return recommendations


class AlchemyBenchSynthesisJudgeSignature(dspy.Signature):
    """
    Expert-level materials scientist signature for evaluating LLM-extracted
    synthesis recipes against reference procedures using AlchemyBench
    methodology.

    This signature implements the 7-criteria evaluation framework from
    AlchemyBench:
    1. Materials Appropriateness (precursors, quantities, purities)
    2. Equipment Appropriateness (apparatus, specifications, parameters)
    3. Procedure Completeness (step organization, detail level)
    4. Procedure Similarity (alignment with ground truth)
    5. Procedure Feasibility (lab execution reality)
    6. Characterization Appropriateness (method selection)
    7. Characterization Similarity (result alignment)

    The evaluation follows expert materials scientist standards with emphasis
    on:
    - Chemical accuracy and stoichiometry
    - Practical laboratory feasibility
    - Safety and handling considerations
    - Equipment availability and specifications
    - Characterization method suitability
    - Result interpretation accuracy
    """

    target_material: str = dspy.InputField(
        description=(
            "Detailed description of the target material to be synthesized, "
            "including composition, structure, and intended properties."
        )
    )

    extracted_recipe: str = dspy.InputField(
        description=(
            "The complete synthesis recipe extracted by an LLM, including "
            "materials, equipment, procedures, and characterization methods."
        )
    )

    reference_procedure: str = dspy.InputField(
        description=(
            "Ground truth synthesis procedure from peer-reviewed scientific "
            "literature, serving as the evaluation standard."
        )
    )

    synthesis_context: str = dspy.InputField(
        description=(
            "Additional context about the synthesis including application "
            "domain, material class, and any special requirements or "
            "constraints."
        )
    )

    evaluation: AlchemyBenchSynthesisEvaluation = dspy.OutputField(
        description=(
            """Comprehensive evaluation following AlchemyBench methodology 
            with structured scoring and detailed reasoning.

EVALUATION APPROACH:
1. Systematically assess each of the 7 criteria on a 1-5 scale (0.5 
   increments allowed)
2. Provide detailed technical reasoning for each score
3. Consider both technical accuracy and practical feasibility
4. Compare directly against the reference procedure
5. Identify critical issues and improvement recommendations

SCORING GUIDELINES:
- 5.0: Excellent - Matches or exceeds reference quality, highly accurate 
  and feasible
- 4.0-4.5: Good - Minor deviations but technically sound and executable
- 3.0-3.5: Adequate - Some issues but generally acceptable approach
- 2.0-2.5: Poor - Significant technical or practical problems
- 1.0-1.5: Very Poor - Major errors, unsafe, or completely unfeasible

FOCUS AREAS:
- Chemical accuracy and stoichiometric correctness
- Equipment specifications and availability
- Procedural safety and laboratory feasibility
- Characterization method appropriateness for target material
- Alignment with established synthesis protocols
- Practical execution considerations and resource requirements

The evaluation must be objective, technically rigorous, and provide 
actionable feedback for synthesis recipe improvement."""
        )
    )


def make_alchemybench_synthesis_judge_signature(
    signature_name: str = "AlchemyBenchSynthesisJudgeSignature",
    instructions: str | None = None,
    target_material_description: str = (
        "Comprehensive description of the target material including "
        "composition, structure, and properties."
    ),
    extracted_recipe_description: str = (
        "Complete LLM-extracted synthesis recipe with all components and "
        "procedures."
    ),
    reference_procedure_description: str = (
        "Ground truth synthesis procedure from scientific literature."
    ),
    synthesis_context_description: str = (
        "Additional synthesis context and requirements."
    ),
    evaluation_description: str = (
        "Comprehensive AlchemyBench-style evaluation with structured scoring."
    ),
) -> type[dspy.Signature]:
    """
    Create an enhanced DSPy signature for AlchemyBench-style synthesis recipe
    evaluation.

    Args:
        signature_name: Name of the signature class
        instructions: Custom instructions (uses AlchemyBench methodology if
            None)
        target_material_description: Description for target material input
        extracted_recipe_description: Description for extracted recipe input
        reference_procedure_description: Description for reference procedure
            input
        synthesis_context_description: Description for synthesis context input
        evaluation_description: Description for evaluation output

    Returns:
        DSPy signature class for synthesis recipe evaluation
    """

    if instructions is None:
        instructions = (
            "You are an expert materials scientist with extensive experience "
            "in synthesis evaluation. Evaluate the extracted synthesis recipe "
            "against the reference procedure using AlchemyBench methodology. "
            "Assess all 7 criteria with detailed technical reasoning and "
            "provide actionable feedback for improvement."
        )

    signature = {
        "target_material": (
            str,
            dspy.InputField(description=target_material_description),
        ),
        "extracted_recipe": (
            str,
            dspy.InputField(description=extracted_recipe_description),
        ),
        "reference_procedure": (
            str,
            dspy.InputField(description=reference_procedure_description),
        ),
        "synthesis_context": (
            str,
            dspy.InputField(description=synthesis_context_description),
        ),
        "evaluation": (
            AlchemyBenchSynthesisEvaluation,
            dspy.OutputField(description=evaluation_description),
        ),
    }

    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
