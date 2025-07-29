from typing import Any

import anthropic


class ClaudeAPIResponse:
    """Response wrapper that includes cost information for Claude API calls."""

    def __init__(
        self,
        content: str,
        cost_usd: float | None = None,
        raw_response: Any = None,
    ):
        self.content = content
        self.cost_usd = cost_usd
        self.raw_response = raw_response


class ClaudeAPIClient:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self._cumulative_cost_usd = 0.0

        if "sonnet" in model_name:
            self.input_cost_per_1m_tokens = 3.00
            self.output_cost_per_1m_tokens = 15.00
        elif "haiku-3" in model_name:
            self.input_cost_per_1m_tokens = 0.25
            self.output_cost_per_1m_tokens = 1.25
        elif "haiku-35" in model_name:
            self.input_cost_per_1m_tokens = 0.80
            self.output_cost_per_1m_tokens = 4.00
        elif "opus" in model_name:
            self.input_cost_per_1m_tokens = 15.00
            self.output_cost_per_1m_tokens = 75.00
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def get_cost(self) -> float:
        """Get the current cumulative cost in USD."""
        return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        """Reset the cumulative cost counter and return the previous value."""
        old_cost = self._cumulative_cost_usd
        self._cumulative_cost_usd = 0.0
        return old_cost

    def _calculate_cost_from_usage(self, response) -> float | None:
        """Calculate cost from Claude API response usage information."""
        try:
            if hasattr(response, "usage") and response.usage:
                # Get token counts
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)

                # Calculate costs
                input_cost = (
                    input_tokens / 1_000_000
                ) * self.input_cost_per_1m_tokens
                output_cost = (
                    output_tokens / 1_000_000
                ) * self.output_cost_per_1m_tokens

                total_cost = input_cost + output_cost
                return total_cost

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

        Returns the text content only.
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

        # Calculate cost from usage information
        cost_usd = self._calculate_cost_from_usage(message)

        # Accumulate cost if available
        if cost_usd is not None:
            self._cumulative_cost_usd += cost_usd

        content = message.content[0].text

        return content
