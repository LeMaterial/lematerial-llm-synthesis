"""Concurrency helpers for LLM API calls: asyncio + semaphore + to_thread.

Used to run synchronous DSPy/LLM code in parallel while capping concurrent
requests to avoid rate limits (429) across providers (Gemini, OpenRouter, etc.).
"""

import asyncio
import os
from collections.abc import Callable
from typing import Any, TypeVar

# Default cap on concurrent LLM calls. Tune down if you hit 429s.
# Override with env LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS (e.g. 8 or 12).
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 10


def get_max_concurrent_llm_calls() -> int:
    """Return max concurrent LLM calls (from env or default)."""
    raw = os.getenv("LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_CONCURRENT_LLM_CALLS


T = TypeVar("T")


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a synchronous function in a thread, holding the semaphore.

    Use this to run sync LLM calls (e.g. extractor.forward, judge.forward)
    in parallel while limiting how many run at once.
    """
    async with semaphore:
        return await asyncio.to_thread(fn, *args, **kwargs)
