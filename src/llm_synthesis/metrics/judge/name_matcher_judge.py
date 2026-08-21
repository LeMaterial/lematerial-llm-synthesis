"""LLM judge for aligning ground-truth names to extracted names.

Mirrors the wiring of ``linking_judge.py`` but performs a much narrower
task: given a list of ground-truth names (materials or plot series) and a
list of LLM/VLM-extracted candidate names, produce a 1:1 alignment that is
tolerant of paraphrase, unit, and notation differences (e.g. "50°C" vs
"323 K", or "Pt/Al2O3" vs "1% Pt supported on alumina").
"""

import json
import logging

import dspy

from llm_synthesis.metrics.judge.base import JudgeInterface
from llm_synthesis.metrics.judge.name_match_ontology import NameMatchResult

logger = logging.getLogger(__name__)


# Input: (gt_names_json, llm_names_json, context)
NameMatcherJudgeInterface = JudgeInterface[
    tuple[str, str, str], NameMatchResult
]


class DspyNameMatcherJudge(NameMatcherJudgeInterface):
    """DSPy module that aligns ground-truth names to extracted names."""

    def __init__(
        self,
        lm: dspy.LM,
        signature: type[dspy.Signature] | None = None,
    ):
        self.signature = signature or NameMatcherJudgeSignature
        self.lm = lm
        super().__init__()

    def forward(self, input: tuple[str, str, str]) -> NameMatchResult:
        gt_names_json, llm_names_json, context = input

        if not gt_names_json or not llm_names_json:
            raise ValueError("gt_names_json and llm_names_json must be set")

        with dspy.settings.context(
            lm=self.lm, adapter=dspy.adapters.JSONAdapter()
        ):
            prediction = dspy.Predict(self.signature)(
                gt_names_json=gt_names_json,
                llm_names_json=llm_names_json,
                context=context,
            )
            return prediction.result


class NameMatcherJudgeSignature(dspy.Signature):
    """Align a ground-truth name list to an extracted-name list."""

    gt_names_json: str = dspy.InputField(
        description="JSON list of ground-truth names, in original order."
    )
    llm_names_json: str = dspy.InputField(
        description="JSON list of candidate extracted names to align against."
    )
    context: str = dspy.InputField(
        description=(
            "Short context describing what these names refer to "
            "(e.g. paper title, material family) to help disambiguate."
        )
    )
    result: NameMatchResult = dspy.OutputField(
        description=(
            """Produce exactly one NameMatchPair per entry in gt_names_json,
in the same order.

For each ground-truth name, find the single best-matching name in
llm_names_json that refers to the same real-world entity, even if phrased
differently. Treat as matches:
- Unit/notation variants (e.g. "50 C" vs "323 K" vs "T=50C")
- Reordered or reformatted composition strings
  (e.g. "Pt/Al2O3" vs "1% Pt supported on alumina" vs "Al2O3-supported Pt")
- LaTeX or unicode formatting differences (e.g. "Al$_2$O$_3$" vs "Al2O3")
- Abbreviations, prefixes, or trailing qualifiers that do not change the
  underlying identity (e.g. sample codes, "_human" suffixes)

Do NOT match names that refer to genuinely different materials, dopant
loadings, or conditions (e.g. "5% Co/MCM-41" is NOT the same as
"10% Co/MCM-41").

Each llm_names_json entry may be used in at most one pair. If no candidate
refers to the same entity as a given ground-truth name, set llm_name to
null for that pair.
"""
        )
    )


def make_name_matcher_judge_signature(
    signature_name: str = "NameMatcherJudgeSignature",
    instructions: str | None = None,
) -> type[dspy.Signature]:
    """Factory for a customised NameMatcherJudge DSPy signature."""
    if instructions is None:
        instructions = (
            "You are aligning two lists of free-text names that describe "
            "the same underlying scientific entities (materials or plot "
            "series) but were produced by different extraction pipelines. "
            "Match names that refer to the same entity despite paraphrase, "
            "unit, or notation differences; do not match genuinely "
            "different entities."
        )

    signature = {
        "gt_names_json": (
            str,
            dspy.InputField(
                description="JSON list of ground-truth names, in order."
            ),
        ),
        "llm_names_json": (
            str,
            dspy.InputField(description="JSON list of candidate names."),
        ),
        "context": (
            str,
            dspy.InputField(description="Disambiguating context."),
        ),
        "result": (
            NameMatchResult,
            dspy.OutputField(description="1:1 name alignment."),
        ),
    }

    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )


def build_name_match_inputs(
    gt_names: list[str], llm_names: list[str], context: str = ""
) -> tuple[str, str, str]:
    """Convenience helper to build the judge's input tuple."""
    return (json.dumps(gt_names), json.dumps(llm_names), context)
