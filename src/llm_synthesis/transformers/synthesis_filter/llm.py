from mistralai import Mistral
from openai import OpenAI, AzureOpenAI
from cohere import ClientV2
import os
from typing import Dict, Optional, Any


class LLMResponse:
    """Response wrapper that includes cost information."""
    
    def __init__(self, content: str, cost_usd: Optional[float] = None, raw_response: Any = None):
        self.content = content
        self.cost_usd = cost_usd
        self.raw_response = raw_response

class LLM: 

    def __init__(self, model_name: str, provider: str, port: int = 8000):
        self.model_name = model_name
        self.provider = provider
        self._cumulative_cost_usd = 0.0

        if self.provider == "mistral":
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        elif self.provider == "cohere":
            self.client = ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        elif self.provider == "openai":
            endpoint = "https://gpt-amayuelas.openai.azure.com/"
            subscription_key = os.getenv("OPENAI_API_KEY")
            api_version = "2024-12-01-preview"
            self.client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key)
        elif self.provider == "vllm":
            self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=os.getenv("VLLM_API_KEY"))
        
    def get_cost(self) -> float:
        """Get the current cumulative cost in USD."""
        return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        """Reset the cumulative cost counter and return the previous value."""
        old_cost = self._cumulative_cost_usd
        self._cumulative_cost_usd = 0.0
        return old_cost

    def _extract_cost_from_response(self, response: Any) -> Optional[float]:
        """Extract cost from API response if available."""
        try:
            # Try to extract cost from usage field (common in OpenAI-style responses)
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'cost'):
                    return float(response.usage.cost)
                # Some providers store cost in different fields
                if hasattr(response.usage, 'total_cost'):
                    return float(response.usage.total_cost)
            
            # Check if response is a dict with usage information
            if isinstance(response, dict):
                usage = response.get('usage', {})
                if 'cost' in usage:
                    return float(usage['cost'])
                if 'total_cost' in usage:
                    return float(usage['total_cost'])
                    
        except (AttributeError, TypeError, ValueError):
            pass
        
        return None

    def generate_text(self, prompt: str, response_format: str = None) -> str:
        """
        Generate text with the LLM and track costs.
        
        Returns the generated text content. Cost tracking is handled internally.
        """
        cost_usd = None
        
        if self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
            
        elif self.provider == "vllm":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
        
        elif self.provider == "cohere":
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.message.content[0].text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
        else:
            raise ValueError(f"Provider {self.provider} not supported")
        
        # Accumulate cost if available
        if cost_usd is not None:
            self._cumulative_cost_usd += cost_usd
            
        return content

    def generate_text_with_cost(self, prompt: str, response_format: str = None) -> LLMResponse:
        """
        Generate text with the LLM and return both content and cost information.
        
        Returns:
            LLMResponse with content and cost_usd fields
        """
        cost_usd = None
        raw_response = None
        
        if self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
            raw_response = response
            
        elif self.provider == "vllm":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
            raw_response = response
        
        elif self.provider == "cohere":
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.message.content[0].text
            raw_response = response
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_usd = self._extract_cost_from_response(response)
            content = response.choices[0].message.content
            raw_response = response
        else:
            raise ValueError(f"Provider {self.provider} not supported")
            
        # Accumulate cost if available
        if cost_usd is not None:
            self._cumulative_cost_usd += cost_usd
            
        return LLMResponse(content=content, cost_usd=cost_usd, raw_response=raw_response)
    