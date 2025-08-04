import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import dspy
import litellm


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str | None = None
    api_base: str | None = None


@dataclass(frozen=True)
class LLMRegistry:
    configs: Mapping[str, LLMConfig]


LLM_REGISTRY = LLMRegistry(
    configs={
        "gemini-2.0-flash": LLMConfig(model="gemini/gemini-2.0-flash"),
        "gemini-2.5-flash": LLMConfig(model="gemini/gemini-2.5-flash-preview-05-20"),
        "gemini-2.5-pro": LLMConfig(model="gemini/gemini-2.5-pro-preview-05-06"),
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
        "local-NuExtract-v1.5": LLMConfig(
            model="openai//scratch16/mshiel10/mzaki4/cache/models--numind--NuExtract-v1.5/snapshots/a7a4e41090a1c5aa95fdebab4c859d7111d628c0",
            api_key="",
            api_base="http://localhost:8000/v1",
        ),
    }
)


class SystemPrefixedLM:
    """
    Wraps a callable LiteLLM-compatible model with an injected system prompt.
    """
    def __init__(self, base_model_callable, system_prompt: str):
        self.base_model_callable = base_model_callable
        self.system_prompt = system_prompt

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt or ""}
            ]
        else:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        return self.base_model_callable(messages=messages, **kwargs)


# Optional auto-injection via env var
if os.getenv("USE_SYSTEM_PREFIXED_LM", "false").lower() == "true":
    config = LLM_REGISTRY.configs["local-NuExtract-v1.5"]
    base_model = dspy.LM(
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key
    )
    wrapped_model = SystemPrefixedLM(
        base_model_callable=base_model,
        system_prompt="You are a materials synthesis expert. Extract structured synthesis from the input paragraph."
    )
    dspy.settings.lm = wrapped_model
