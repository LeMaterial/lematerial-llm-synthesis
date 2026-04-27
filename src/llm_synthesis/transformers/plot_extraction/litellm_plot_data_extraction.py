"""Generic plot data extractor using litellm (supports any vision model)."""

import logging
import re

import litellm

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import ExtractedLinePlotData
from llm_synthesis.transformers.plot_extraction.base import (
    LinePlotDataExtractorInterface,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction import (
    resources,
)


class LiteLLMPlotDataExtractor(LinePlotDataExtractorInterface):
    """Plot data extractor using litellm — works with any vision model.

    Uses the same prompt and parsing logic as ClaudeLinePlotDataExtractor,
    but routes API calls through litellm for multi-provider support.
    """

    def __init__(
        self,
        model: str,
        prompt: str = resources.LINE_CHART_PROMPT_WITH_CONTEXT,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_kwargs: dict | None = None,
        retry_temperatures: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.extra_kwargs = extra_kwargs or {}
        self.retry_temperatures = retry_temperatures or [temperature, 0.3, 0.5]
        self._cumulative_cost_usd = 0.0

    def forward(self, input: FigureInfoWithPaper) -> ExtractedLinePlotData:
        figure_base64 = input.base64_data

        # Build prompt with figure context
        figure_context = f"{input.context_before}\n{input.context_after}"
        prompt = self.prompt.format(figure_context=figure_context)

        # Detect image type
        image_type = "jpeg" if figure_base64.startswith("/9j/") else "png"

        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/{image_type}"
                                    f";base64,{figure_base64}"
                                )
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        for k, v in self.extra_kwargs.items():
            if k not in ("thinking", "reasoning_effort", "enable_thinking"):
                kwargs[k] = v

        last_exc: Exception | None = None
        for t_idx, temp in enumerate(self.retry_temperatures):
            kwargs["temperature"] = temp
            try:
                response = litellm.completion(**kwargs)

                # Track cost
                try:
                    cost = litellm.completion_cost(completion_response=response)
                    self._cumulative_cost_usd += cost
                except Exception:
                    pass

                response_text = response.choices[0].message.content
                return self._parse_into_pydantic(response_text)

            except Exception as e:
                last_exc = e
                if t_idx < len(self.retry_temperatures) - 1:
                    logging.warning(
                        "VLM extractor: failure at temp=%.1f: %r"
                        " — retrying at temp=%.1f",
                        temp, e, self.retry_temperatures[t_idx + 1],
                    )
                else:
                    logging.warning(
                        "VLM extractor: all temperatures exhausted: %r", e
                    )
        raise last_exc

    def _parse_into_pydantic(self, response: str) -> ExtractedLinePlotData:
        """Parse VLM response text into structured plot data.

        Same logic as ClaudeLinePlotDataExtractor._parse_into_pydantic.
        """
        lines = response.strip().split("\n")

        data = {
            "name_to_coordinates": {},
            "title": None,
            "x_axis_label": None,
            "x_axis_unit": None,
            "y_left_axis_label": None,
            "y_left_axis_unit": None,
        }

        metadata_patterns = {
            "title": re.compile(r"^title:\s*(.*)$"),
            "x_axis_label": re.compile(r"^x_axis_label:\s*(.*)$"),
            "x_axis_unit": re.compile(r"^x_axis_unit:\s*(.*)$"),
            "y_left_axis_label": re.compile(r"^y_left_axis_label:\s*(.*)$"),
            "y_left_axis_unit": re.compile(r"^y_left_axis_unit:\s*(.*)$"),
        }

        line_pattern = re.compile(r"^(.*?):\s*\[\[(.*?)\]\]$")

        for line in lines:
            line = line.strip()

            if match := line_pattern.match(line):
                name, coords_str = match.groups()
                coords = [
                    list(map(float, coord.split(",")))
                    for coord in coords_str.split("], [")
                ]
                data["name_to_coordinates"][name] = coords
                continue

            for key, pattern in metadata_patterns.items():
                if match := pattern.match(line):
                    data[key] = match.group(1).strip()
                    break

        return ExtractedLinePlotData(**data)

    def get_cost(self) -> float:
        return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        old = self._cumulative_cost_usd
        self._cumulative_cost_usd = 0.0
        return old
