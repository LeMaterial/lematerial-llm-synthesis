"""Generic plot data extractor using litellm (supports any vision model)."""

import logging
import re
import threading

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
        max_tokens: int = 8192,
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
        # This extractor instance is shared across all worker threads in
        # run_from_hf.py's ThreadPoolExecutor (built once in
        # build_pipeline()) -- without a lock, concurrent += on a shared
        # float silently drops updates and per-paper before/after cost
        # snapshots pick up other papers' concurrent cost, both of which
        # were observed inflating per-paper cost readings ~10x under
        # --workers 12.
        self._cost_lock = threading.Lock()

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
                    with self._cost_lock:
                        self._cumulative_cost_usd += cost
                except Exception:
                    pass

                response_text = response.choices[0].message.content
                if response_text is None:
                    raise ValueError("VLM returned None content")
                return self._parse_into_pydantic(response_text)

            except Exception as e:
                last_exc = e
                if t_idx < len(self.retry_temperatures) - 1:
                    logging.warning(
                        "VLM extractor: failure at temp=%.1f: %r"
                        " — retrying at temp=%.1f",
                        temp,
                        e,
                        self.retry_temperatures[t_idx + 1],
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

        # Single-line: "Name: [[x,y], ...]"  (Claude style)
        line_pattern = re.compile(r"^(.*?):\s*\[\[(.*)\]\]\s*$")
        # Standalone coords line: "[[x,y], ...]"  (Gemini, follows name line)
        coords_only_pattern = re.compile(r"^\[\[(.*)\]\]\s*$")
        # "Series_Name: actual name"  (Gemini style label line)
        series_label_pattern = re.compile(
            r"^Series_Name:\s*(.+)$", re.IGNORECASE
        )

        pending_name: str | None = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Standalone coords: consume if we have a pending name
            if m := coords_only_pattern.match(line):
                if pending_name is not None:
                    coords_str = m.group(1)
                    coords = [
                        list(map(float, c.split(",")))
                        for c in coords_str.split("], [")
                    ]
                    data["name_to_coordinates"][pending_name] = coords
                    pending_name = None
                    continue
                # no pending name — ignore orphan coords line
                continue

            pending_name = None

            # Single-line format
            if m := line_pattern.match(line):
                name, coords_str = m.groups()
                coords = [
                    list(map(float, c.split(",")))
                    for c in coords_str.split("], [")
                ]
                data["name_to_coordinates"][name] = coords
                continue

            # Gemini "Series_Name: <actual name>" label line
            if m := series_label_pattern.match(line):
                pending_name = m.group(1).strip()
                continue

            # Metadata
            for key, pattern in metadata_patterns.items():
                if m := pattern.match(line):
                    data[key] = m.group(1).strip()
                    break

        return ExtractedLinePlotData(**data)

    def get_cost(self) -> float:
        with self._cost_lock:
            return self._cumulative_cost_usd

    def reset_cost(self) -> float:
        with self._cost_lock:
            old = self._cumulative_cost_usd
            self._cumulative_cost_usd = 0.0
        return old
