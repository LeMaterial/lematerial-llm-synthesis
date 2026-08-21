"""Pydantic ontology for LLM-judged name-alignment matching.

Used to align two lists of free-text names (e.g. material names or plot
series names) that refer to the same underlying entities but are phrased
differently by different extraction pipelines.
"""

from typing import Literal

from pydantic import BaseModel, Field


class NameMatchPair(BaseModel):
    """One aligned pair between a ground-truth name and an extracted name."""

    gt_name: str = Field(description="The ground-truth name being aligned.")
    llm_name: str | None = Field(
        description=(
            "The extracted name judged to refer to the same entity as "
            "gt_name, or null if no candidate refers to the same entity."
        )
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence that gt_name and llm_name are the same entity."
    )
    reasoning: str = Field(
        description="Brief justification for the match (or non-match)."
    )


class NameMatchResult(BaseModel):
    """Full 1:1 alignment between a ground-truth and an extracted name list."""

    matches: list[NameMatchPair] = Field(
        description=(
            "One entry per ground-truth name, in the same order as the "
            "input gt_names list. Each extracted name may be used in at "
            "most one pair."
        )
    )
