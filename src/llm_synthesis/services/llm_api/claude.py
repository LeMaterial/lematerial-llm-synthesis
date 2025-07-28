import anthropic
from typing import Optional, Dict, Any


class ClaudeAPIResponse:
    """Response wrapper that includes cost information for Claude API calls."""
    
    def __init__(self, content: str, cost_usd: Optional[float] = None, raw_response: Any = None):
        self.content = content
        self.cost_usd = cost_usd
        self.raw_response = raw_response


class ClaudeAPIClient:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self._cumulative_cost_usd = 0.0

    def get_cost(self) -> float:
        """Get the current cumulative cost in USD."""
        return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        """Reset the cumulative cost counter and return the previous value."""
        old_cost = self._cumulative_cost_usd
        self._cumulative_cost_usd = 0.0
        return old_cost

    def _extract_cost_from_response(self, response) -> Optional[float]:
        """Extract cost from Claude API response if available."""
        try:
            # Claude may provide usage information in the response
            if hasattr(response, 'usage') and response.usage:
                # Check for various cost fields
                for cost_field in ['cost', 'total_cost', 'prompt_cost', 'completion_cost']:
                    if hasattr(response.usage, cost_field):
                        cost_value = getattr(response.usage, cost_field)
                        if cost_value is not None:
                            return float(cost_value)
            
            # Some responses may have billing information in metadata
            if hasattr(response, 'billing') and response.billing:
                if hasattr(response.billing, 'cost_usd'):
                    return float(response.billing.cost_usd)
                    
        except (AttributeError, TypeError, ValueError):
            pass
        
        return None

    def vision_model_api_call(
        self,
        figure_base64: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Note: Claude API call can very quickly reach the token limit.
        If we want to batch process images, we should think carefully
        how to handle retry to not receive excessive bills.
        
        Returns the text content only. Use vision_model_api_call_with_cost
        for cost information.
        """
        result = self.vision_model_api_call_with_cost(
            figure_base64, prompt, max_tokens, temperature
        )
        return result.content

    def vision_model_api_call_with_cost(
        self,
        figure_base64: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ClaudeAPIResponse:
        """
        Make a vision API call and return both content and cost information.
        
        Returns:
            ClaudeAPIResponse with content and cost_usd fields
        """
        image_type = "jpeg" if figure_base64.startswith("/9j/") else "png"
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/" + image_type,
                                "data": figure_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        
        # Extract cost information
        cost_usd = self._extract_cost_from_response(message)
        
        # Accumulate cost if available
        if cost_usd is not None:
            self._cumulative_cost_usd += cost_usd
            
        content = message.content[0].text
        
        return ClaudeAPIResponse(content=content, cost_usd=cost_usd, raw_response=message)
