import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import dspy


@dataclass(frozen=True)
class LLMConfig:
    """
    A configuration for an LLM to instantiate with dspy.
    Includes the model name, and optional API key name in the
    environment (e.g. "OPENAI_API_KEY") and base URL.
    The latter is needed to call external providers with the OpenAI API.
    In DSPy, you can use dozens of LLM providers supported by LiteLLM.
    Simply follow their instructions for which {PROVIDER}_API_KEY to set and
    how to write pass the {provider_name}/{model_name} to the constructor.

    Args:
        model: The name of the model to instantiate.
        api_key: The name of the environment variable containing the API key.
        api_base: The base URL of the API.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None


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
        "gemini-2.5-flash": LLMConfig(
            model="gemini/gemini-2.5-flash-preview-05-20"
        ),
        "gemini-2.5-pro": LLMConfig(
            model="gemini/gemini-2.5-pro-preview-05-06"
        ),
        "gpt-4o": LLMConfig(model="openai/gpt-4o"),
        "gpt-4o-mini": LLMConfig(model="openai/gpt-4o-mini"),
        "gpt-o4-mini": LLMConfig(model="openai/o4-mini-2025-04-16"),
        "gpt-o3-mini": LLMConfig(model="openai/o3-mini-2025-01-31"),
        "gpt-4.1": LLMConfig(model="openai/gpt-4.1-2025-04-14"),
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


class SystemPrefixedLM:
    """
    Wrap any dspy.LM and automatically inject a system prompt
    at start of every call.
    """

    def __init__(self, system_prompt: str, llm: dspy.LM):
        """
        Wrap any dspy.LM and automatically inject a system prompt
        at start of every call.

        Args:
            system_prompt: prompt to inject at start of every call.
            llm: The dspy.LM to wrap.
        """
        self._system_prompt = system_prompt
        self._llm = llm

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """
        Call wrapped dspy.LM with system prompt injected
        at start of every call.

        Args:
            prompt: The prompt to inject at the start of every call.
            messages: The messages to inject at start of every call.
        """
        if messages is None:
            # user passed raw prompt, turn into a 2-message chat
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt or ""},
            ]
        else:
            # chat-style already, just prepend
            messages = [
                {"role": "system", "content": self._system_prompt},
                *messages,
            ]

        # delegate to the real LM
        return self._llm(messages=messages, **kwargs)

    def __getattr__(self, attr):
        # proxy everything else back to the underlying LM
        return getattr(self._llm, attr)
