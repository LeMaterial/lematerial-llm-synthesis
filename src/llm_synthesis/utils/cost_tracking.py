"""
Cost tracking utilities for LLM calls in the lematerial-llm-synthesis project.
"""

from dataclasses import dataclass
from typing import Any

import dspy


@dataclass
class CostAwareResponse:
    """Base class for responses that include cost information."""
    cost_usd: float | None = None
    raw_response: Any = None


@dataclass 
class DSPyResponseWithCost:
    """Wrapper for DSPy responses that includes cost information."""
    response: Any
    cost_usd: float | None = None
    cumulative_cost_usd: float | None = None
    

def extract_cost_from_dspy_response(response: Any) -> float | None:
    """
    Extract cost information from a DSPy response using multiple fallback methods.
    
    Args:
        response: The DSPy response object to extract cost from
        
    Returns:
        Cost in USD as a float, or None if not available
    """
    try:
        # Method 1: Check DSPy prediction's LM usage
        if hasattr(response, 'get_lm_usage'):
            lm_usage = response.get_lm_usage()
            if lm_usage and isinstance(lm_usage, dict):
                # Check for cost in usage data
                for cost_key in ['cost', 'response_cost', 'total_cost']:
                    if cost_key in lm_usage and lm_usage[cost_key] is not None:
                        return float(lm_usage[cost_key])
        
        # Method 2: Check LM usage attribute directly  
        if hasattr(response, '_lm_usage'):
            lm_usage = response._lm_usage
            if lm_usage and isinstance(lm_usage, dict):
                for cost_key in ['cost', 'response_cost', 'total_cost']:
                    if cost_key in lm_usage and lm_usage[cost_key] is not None:
                        return float(lm_usage[cost_key])
        
        # Method 3: Check current DSPy LM history for the most recent entry
        try:
            import dspy
            if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
                history = dspy.settings.lm.history
                if history:
                    # Get the most recent entry
                    last_entry = history[-1]
                    if isinstance(last_entry, dict) and 'cost' in last_entry:
                        cost = last_entry['cost']
                        if cost is not None:
                            return float(cost)
        except (ImportError, AttributeError):
            pass
        
        # Method 4: Check for raw response data (LiteLLM)
        if hasattr(response, '_raw'):
            raw_response = response._raw
            
            # Check for LiteLLM response_cost in raw response
            if (hasattr(raw_response, 'response_cost') and 
                raw_response.response_cost is not None):
                return float(raw_response.response_cost)
            
            # Check usage dictionary
            usage = raw_response.get('usage', {})
            if isinstance(usage, dict):
                for cost_key in ['cost', 'total_cost', 'response_cost']:
                    cost = usage.get(cost_key)
                    if cost is not None:
                        return float(cost)
            
            # Check top-level cost fields in raw response
            if isinstance(raw_response, dict):
                for cost_key in ['cost', 'response_cost', 'total_cost']:
                    cost = raw_response.get(cost_key)
                    if cost is not None:
                        return float(cost)
        
        # Method 5: Check if response has usage tracking built-in
        if hasattr(response, 'get_usage_and_cost'):
            usage_data = response.get_usage_and_cost()
            if isinstance(usage_data, dict):
                for cost_key in ['cost', 'response_cost', 'total_cost']:
                    cost = usage_data.get(cost_key)
                    if cost is not None:
                        return float(cost)
        
        # Method 6: Additional fallback - check direct cost attributes on response
        for cost_attr in ['cost_usd', 'cost', 'total_cost']:
            if hasattr(response, cost_attr):
                cost = getattr(response, cost_attr)
                if cost is not None:
                    return float(cost)
                
    except (AttributeError, TypeError, ValueError):
        # If cost extraction fails, return None
        pass
    
    return None


def get_cumulative_cost_from_lm(lm: dspy.LM) -> float | None:
    """
    Get cumulative cost from a DSPy language model if it supports cost tracking.
    
    Args:
        lm: DSPy language model instance
        
    Returns:
        Cumulative cost in USD, or None if not available
    """
    if hasattr(lm, 'get_cost'):
        return lm.get_cost()
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
        if lm and hasattr(lm, 'get_cost'):
            lm_cost = lm.get_cost()
            if lm_cost is not None and lm_cost > self._session_cost_usd:
                # Assume the difference is from this call
                call_cost = lm_cost - self._session_cost_usd
                self._session_cost_usd = lm_cost
                return call_cost
        
        return None


def create_cost_aware_response(
    original_response: Any, cost_usd: float | None = None
) -> DSPyResponseWithCost:
    """
    Create a cost-aware wrapper for any response object.
    
    Args:
        original_response: The original response object
        cost_usd: Cost in USD for this response
        
    Returns:
        DSPyResponseWithCost wrapper
    """
    return DSPyResponseWithCost(
        response=original_response,
        cost_usd=cost_usd
    )


def create_batch_cost_summary(
    responses: list[Any], individual_costs: list[float] | None = None
) -> dict[str, Any]:
    """
    Create a cost summary for batch operations.
    
    Args:
        responses: List of response objects
        individual_costs: Optional list of individual costs 
                         (will be extracted if not provided)
        
    Returns:
        Dictionary with batch cost summary
    """
    if individual_costs is None:
        individual_costs = [
            extract_cost_from_dspy_response(response) 
            for response in responses
        ]
    
    # Calculate total cost (only for non-None values)
    valid_costs = [cost for cost in individual_costs if cost is not None]
    total_cost = sum(valid_costs) if valid_costs else None
    
    return {
        "individual_costs_usd": individual_costs,
        "total_cost_usd": total_cost,
        "num_requests": len(responses),
        "num_requests_with_cost": len(valid_costs)
    }


# Export cost tracking utilities in dspy_utils
def add_cost_tracking_to_dspy_utils():
    """Add cost tracking utilities to the main dspy_utils module."""
    try:
        from llm_synthesis.utils import dspy_utils
        
        # Add these functions to dspy_utils module
        dspy_utils.extract_cost_from_dspy_response = (
            extract_cost_from_dspy_response
        )
        dspy_utils.get_cumulative_cost_from_lm = get_cumulative_cost_from_lm
        dspy_utils.create_cost_aware_response = create_cost_aware_response
        dspy_utils.create_batch_cost_summary = create_batch_cost_summary
        dspy_utils.DSPyResponseWithCost = DSPyResponseWithCost
        dspy_utils.CostTrackingMixin = CostTrackingMixin
        
    except ImportError:
        # If dspy_utils can't be imported, just continue
        pass


# Automatically add to dspy_utils when this module is imported
add_cost_tracking_to_dspy_utils()
