"""
Cost tracking utilities for LLM calls in the lematerial-llm-synthesis project.
"""

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


def extract_cost_from_dspy_response(response: Any) -> float | None:
    """
    Extract cost information from DSPy response using multiple fallback methods.

    Args:
        response: The DSPy response object to extract cost from

    Returns:
        Cost in USD as a float, or None if not available
    """
    try:
        if hasattr(dspy.settings, "lm") and hasattr(
            dspy.settings.lm, "history"
        ):
            history = dspy.settings.lm.history
            if history:
                # Get the most recent entry
                last_entry = history[-1]
                if isinstance(last_entry, dict) and "cost" in last_entry:
                    cost = last_entry["cost"]
                    if cost is not None:
                        return float(cost)

    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("cost extraction failed: %r", exc)

    return None
