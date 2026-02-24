"""LLM-based series-to-material linking transformer."""

import json
import logging
import re

import dspy

from llm_synthesis.models.performance import SeriesMapping
from llm_synthesis.transformers.performance_linking.base import (
    LinkingInput,
    PerformanceLinkingInterface,
)
from llm_synthesis.utils.formula_utils import normalize_formula

logger = logging.getLogger(__name__)


DEFAULT_MATCHING_PROMPT = """You are given materials studied in a scientific paper and series/line names
extracted from a plot in that paper. Match each series name to the material
it represents.

Materials: {materials}
Series names from plot: {series_names}
Figure context: {context}
Plot title: {plot_title}
X-axis: {x_axis_label} ({x_axis_unit})
Y-axis: {y_axis_label} ({y_axis_unit})

Return a JSON list of matches. Each match must have:
- "series_name": exactly one of the series names listed above
- "material_name": exactly one of the material names listed above
- "confidence": "high", "medium", or "low"
- "reasoning": 1-2 WORDS only (e.g. "formula match", "composition match")

Rules:
- Only match when there is a CLEAR connection between the series name and a material name
  (e.g., the names are similar, or context makes the link obvious).
- NEVER guess or assign matches randomly. If you cannot identify a real connection, leave
  the series unmatched. An empty list [] is a valid response.
- If a series is a baseline, reference, substrate, equilibrium line, or pressure condition — do NOT include it.
- series_name and material_name MUST be exactly from the lists above. Do not modify them.
- Formatting differences are irrelevant: "Re0.88Mo0.12" and "Re_{{0.88}}Mo_{{0.12}}" are the SAME material.
  Match based on chemical composition, not string formatting.
- KEEP reasoning extremely short to avoid response truncation.

Return ONLY a valid JSON list, no other text."""


class SeriesMaterialLinker(PerformanceLinkingInterface):
    """LLM-based transformer for matching plot series names to material names.

    This transformer uses an LLM to semantically match series names from plots
    (e.g., "575", "Ni/Al2O3", "Sample A") to the actual material names extracted
    from the paper (e.g., "Mo2(C,N)Tx-575", "10%Ni/Al2O3").

    Attributes:
        lm: DSPy language model for making predictions
        prompt_template: Template for the matching prompt
    """

    def __init__(
        self,
        lm: dspy.LM,
        prompt_template: str = DEFAULT_MATCHING_PROMPT,
    ):
        """Initialize the linker.

        Args:
            lm: DSPy language model instance
            prompt_template: Prompt template with placeholders for materials,
                series_names, context, plot_title, x_axis_label, x_axis_unit,
                y_axis_label, y_axis_unit
        """
        super().__init__()
        self.lm = lm
        self.prompt_template = prompt_template

    def forward(self, input: LinkingInput) -> list[SeriesMapping]:
        """Match plot series names to material names using LLM.

        Args:
            input: LinkingInput containing materials, series names, context,
                and plot metadata

        Returns:
            List of validated SeriesMapping objects
        """
        prompt = self._build_prompt(input)

        # Call LLM
        response = self.lm(prompt)
        response_text = response if isinstance(response, str) else response[0]

        if not response_text:
            logger.warning("LLM returned empty/None response (possibly truncated)")
            return []

        # Parse response
        raw_mappings = self._parse_response(response_text)

        # Validate mappings
        validated = self._validate_mappings(
            raw_mappings,
            input.series_names,
            input.materials,
        )

        return validated

    def _build_prompt(self, input: LinkingInput) -> str:
        """Build the matching prompt from input."""
        return self.prompt_template.format(
            materials=json.dumps(input.materials),
            series_names=json.dumps(input.series_names),
            context=input.context,
            plot_title=input.plot_metadata.get("title", "N/A"),
            x_axis_label=input.plot_metadata.get("x_axis_label", "N/A"),
            x_axis_unit=input.plot_metadata.get("x_axis_unit", ""),
            y_axis_label=input.plot_metadata.get("y_left_axis_label", "N/A"),
            y_axis_unit=input.plot_metadata.get("y_left_axis_unit", ""),
        )

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse LLM response into list of mapping dicts."""
        response_text = response_text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        # Try to extract JSON array from response if there's extra text
        if "[" in response_text and "]" in response_text:
            start = response_text.index("[")
            end = response_text.rindex("]") + 1
            response_text = response_text[start:end]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract individual JSON objects from malformed array
            objects = re.findall(r"\{[^{}]+\}", response_text)
            results = []
            for obj_str in objects:
                try:
                    results.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    continue
            if results:
                logger.debug(f"Recovered {len(results)} mappings from malformed JSON")
                return results
            logger.warning(
                f"Failed to parse LLM response.\nRaw: {response_text[:300]}"
            )
            return []

    def _validate_mappings(
        self,
        raw_mappings: list[dict],
        valid_series: list[str],
        valid_materials: list[str],
    ) -> list[SeriesMapping]:
        """Validate mappings against known series and materials.

        Uses fuzzy formula normalization as a fallback when exact string
        matching fails. This handles cases where the LLM returns
        "Re0.88Mo0.12" but the valid material is "Re_{0.88}Mo_{0.12}".

        Args:
            raw_mappings: Raw mapping dicts from LLM
            valid_series: List of valid series names from the plot
            valid_materials: List of valid material names from extraction

        Returns:
            List of validated SeriesMapping objects
        """
        valid_series_set = set(valid_series)
        valid_materials_set = set(valid_materials)

        # Build normalized lookup for fuzzy fallback
        norm_to_series: dict[str, str] = {}
        for s in valid_series:
            norm = normalize_formula(s)
            if norm not in norm_to_series:
                norm_to_series[norm] = s

        norm_to_material: dict[str, str] = {}
        for mat in valid_materials:
            norm = normalize_formula(mat)
            if norm not in norm_to_material:
                norm_to_material[norm] = mat

        validated = []

        for m in raw_mappings:
            sn = m.get("series_name", "")
            mn = m.get("material_name", "")

            # --- Validate series_name ---
            if sn not in valid_series_set:
                sn_norm = normalize_formula(sn)
                resolved_sn = norm_to_series.get(sn_norm)
                if resolved_sn:
                    logger.debug(f"Fuzzy series match: '{sn}' -> '{resolved_sn}'")
                    sn = resolved_sn
                else:
                    logger.debug(f"Discarding mapping: series '{sn}' not in plot")
                    continue

            # --- Validate material_name ---
            if mn not in valid_materials_set:
                mn_norm = normalize_formula(mn)
                resolved_mn = norm_to_material.get(mn_norm)
                if resolved_mn:
                    logger.debug(f"Fuzzy material match: '{mn}' -> '{resolved_mn}'")
                    mn = resolved_mn
                else:
                    logger.debug(
                        f"Discarding mapping: material '{mn}' not in list"
                    )
                    continue

            validated.append(
                SeriesMapping(
                    series_name=sn,
                    material_name=mn,
                    confidence=m.get("confidence", "medium"),
                    reasoning=m.get("reasoning", ""),
                )
            )

        return validated
