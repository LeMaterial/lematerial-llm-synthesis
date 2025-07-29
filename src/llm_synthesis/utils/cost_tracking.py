"""
Cost tracking utilities for LLM calls in the lematerial-llm-synthesis project.
"""

from typing import Any

import dspy


def extract_cost_from_dspy_response() -> float | None:
    """
    Extract cost information from DSPy response.

    Args:
        response: The DSPy response object to extract cost from

    Returns:
        Cost in USD as a float, or None if not available
    """
    if hasattr(dspy.settings, "lm") and hasattr(dspy.settings.lm, "history"):
        history = dspy.settings.lm.history
        if history:
            # Get the most recent entry
            last_entry = history[-1]
            if isinstance(last_entry, dict) and "cost" in last_entry:
                cost = last_entry["cost"]
                if cost is not None:
                    return float(cost)

    return None


class CostTrackingMixin:
    """
    Mixin class that adds cost tracking capabilities to DSPy-based extractors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_cost_usd = 0.0

    def get_session_cost(self) -> float:
        """Get the cost accumulated in this session."""
        return self._session_cost_usd

    def reset_session_cost(self) -> float:
        """Reset session cost and return the previous value."""
        old_cost = self._session_cost_usd
        self._session_cost_usd = 0.0
        return old_cost

    def _track_response_cost(
        self, response: Any, lm: dspy.LM | None = None
    ) -> float | None:
        """
        Track cost from a response and optionally from the language model.

        Args:
            response: The response to extract cost from
            lm: Optional language model to get cumulative cost from

        Returns:
            Cost for this specific response, or None if not available
        """
        # Extract cost from the response
        response_cost = extract_cost_from_dspy_response(response)

        if response_cost is not None:
            self._session_cost_usd += response_cost
            return response_cost

        # If response cost is not available, try to get it from the LM
        if lm and hasattr(lm, "get_cost"):
            lm_cost = lm.get_cost()
            if lm_cost is not None and lm_cost > self._session_cost_usd:
                # Assume the difference is from this call
                call_cost = lm_cost - self._session_cost_usd
                self._session_cost_usd = lm_cost
                return call_cost

        return None


# Export cost tracking utilities in dspy_utils
def add_cost_tracking_to_dspy_utils():
    """Add cost tracking utilities to the main dspy_utils module."""
    try:
        from llm_synthesis.utils import dspy_utils

        # Add these functions to dspy_utils module
        dspy_utils.extract_cost_from_dspy_response = (
            extract_cost_from_dspy_response
        )
        dspy_utils.CostTrackingMixin = CostTrackingMixin

    except ImportError:
        # If dspy_utils can't be imported, just continue
        pass


# Automatically add to dspy_utils when this module is imported
add_cost_tracking_to_dspy_utils()
