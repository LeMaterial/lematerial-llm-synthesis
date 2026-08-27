"""
Cost tracking utilities for LLM calls in the lematerial-llm-synthesis project.
"""

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


def extract_cost_from_dspy_response(
    response: Any, lm: dspy.LM | None = None
) -> float | None:
    """
    Extract cost information from DSPy response using multiple fallback methods.

    Args:
        response: The DSPy response object to extract cost from
        lm: The specific LM instance the call was made on. Preferred over
            dspy.settings.lm, which reflects whatever LM is globally
            configured (via dspy.settings.configure) and may not match the
            instance actually used -- e.g. when an LM is constructed and
            passed directly to an extractor without a matching global
            dspy.settings.configure() call, dspy.settings.lm.history would
            silently not contain this call's entry at all.

    Returns:
        Cost in USD as a float, or None if not available
    """
    candidates = []
    if lm is not None and hasattr(lm, "history"):
        candidates.append(lm.history)
    if hasattr(dspy.settings, "lm") and hasattr(dspy.settings.lm, "history"):
        candidates.append(dspy.settings.lm.history)

    for history in candidates:
        try:
            if history:
                last_entry = history[-1]
                if isinstance(last_entry, dict) and "cost" in last_entry:
                    cost = last_entry["cost"]
                    if cost is not None:
                        return float(cost)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("cost extraction failed: %r", exc)

    return None
