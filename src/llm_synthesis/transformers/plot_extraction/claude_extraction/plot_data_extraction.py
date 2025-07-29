import re

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import ExtractedLinePlotData
from llm_synthesis.services.llm_api.claude import (
    ClaudeAPIClient,
)
from llm_synthesis.transformers.plot_extraction.base import (
    LinePlotDataExtractorInterface,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction import (
    resources,
)
from llm_synthesis.utils.cost_tracking import CostTrackingMixin


class ClaudeLinePlotDataExtractor(
    LinePlotDataExtractorInterface, CostTrackingMixin
):
    def __init__(
        self,
        model_name: str,
        prompt: str = resources.LINE_CHART_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        super().__init__()
        CostTrackingMixin.__init__(self)
        self.claude_client = ClaudeAPIClient(model_name)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def forward(
        self,
        input: FigureInfoWithPaper,
    ) -> ExtractedLinePlotData:
        figure_base64 = input.base64_data

        # Use the cost-aware method
        claude_response_obj = (
            self.claude_client.vision_model_api_call_with_cost(
                figure_base64=figure_base64,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        )

        # Track the cost if available
        if claude_response_obj.cost_usd is not None:
            self._session_cost_usd += claude_response_obj.cost_usd

        return self._parse_into_pydantic(claude_response_obj.content)

    def get_cost(self) -> float:
        """Get cumulative cost from both Claude client and session tracking."""
        claude_cost = self.claude_client.get_cost()
        session_cost = self._session_cost_usd
        return max(claude_cost, session_cost)  # Return the higher value

    def reset_cost(self) -> float:
        """Reset costs in both Claude client and session tracker."""
        claude_cost = self.claude_client.reset_cost()
        session_cost = self.reset_session_cost()
        return max(claude_cost, session_cost)

    def _parse_into_pydantic(self, response: str) -> ExtractedLinePlotData:
        """
        Parse text into Pydantic object with regex pattern matching
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
