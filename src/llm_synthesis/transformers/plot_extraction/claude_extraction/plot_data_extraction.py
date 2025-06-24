import re

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import ClaudeExtractedPlotData
from llm_synthesis.transformers.plot_extraction.claude_extraction import (
    resources,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.claude import (
    ClaudeAPIClient,
)


class ClaudePlotDataExtractor:
    def __init__(self, model_name: str):
        self.claude_client = ClaudeAPIClient(model_name)

    def extract(
        self,
        input: FigureInfoWithPaper,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ClaudeExtractedPlotData:
        figure_base64 = input.base64_data
        claude_response = self.claude_client.vision_model_api_call(
            figure_base64=figure_base64,
            prompt=resources.LINE_CHART_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._parse_into_pydantic(claude_response)

    def _parse_into_pydantic(self, response: str) -> ClaudeExtractedPlotData:
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

        return ClaudeExtractedPlotData(**data)
