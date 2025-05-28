import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class LLMConfig:
    """
    A configuration for an LLM to instantiate with dspy. Includes the model name, and optional API key name in the environment (e.g. "OPENAI_API_KEY") and base URL. The latter is needed to call external providers with the OpenAI API. In DSPy, you can use any of the dozens of LLM providers supported by LiteLLM. Simply follow their instructions for which {PROVIDER}_API_KEY to set and how to write pass the {provider_name}/{model_name} to the constructor.

    Args:
        model: The name of the model to instantiate.
        api_key: The name of the environment variable containing the API key.
        api_base: The base URL of the API.
    """

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass(frozen=True)
class LLMRegistry:
    """
    A registry of LLMs to instantiate with dspy.

    Args:
        configs: A mapping of model names to LLM configurations.
    """

    configs: Mapping[str, LLMConfig]


LLM_REGISTRY = LLMRegistry(
    configs={
        "gemini-2.0-flash": LLMConfig(model="gemini/gemini-2.0-flash"),
        "gemini-2.5-flash": LLMConfig(model="gemini/gemini-2.5-flash-preview-05-20"),
        "gemini-2.5-pro": LLMConfig(model="gemini/gemini-2.5-pro-preview-05-06"),
        "gpt-4o": LLMConfig(model="openai/gpt-4o"),
        "gpt-4o-mini": LLMConfig(model="openai/gpt-4o-mini"),
        "mistral-small": LLMConfig(
            model="openai/mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
        "mistral-medium": LLMConfig(
            model="openai/mistral-medium-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
        "mistral-large": LLMConfig(
            model="openai/mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            api_base="https://api.mistral.ai/v1/",
        ),
    }
)
