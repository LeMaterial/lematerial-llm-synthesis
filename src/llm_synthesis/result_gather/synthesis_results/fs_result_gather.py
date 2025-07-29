import json
import os
from datetime import datetime
from typing import Any

import fsspec

from llm_synthesis.models.paper import PaperWithSynthesisOntologies
from llm_synthesis.result_gather.base import ResultGatherInterface


class SynthesisFSResultGather(
    ResultGatherInterface[PaperWithSynthesisOntologies]
):
    def __init__(self, result_dir: str = ""):
        self.result_dir = result_dir
        self.fs, _, _ = fsspec.get_fs_token_paths(self.result_dir)
        self._ensure_dir(self.result_dir)

    def gather(
        self,
        paper: PaperWithSynthesisOntologies,
        cost_data: dict[str, Any] | None = None,
    ):
        self._ensure_dir(os.path.join(self.result_dir, paper.id))

        # Save the main synthesis (first material's synthesis)
        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "result.json"), "w"
        ) as f:
            if paper.all_syntheses:
                f.write(
                    json.dumps(
                        [
                            synthesis.model_dump()
                            for synthesis in paper.all_syntheses
                        ],
                        indent=2,
                    )
                )
            else:
                f.write(json.dumps({"error": "No synthesis found"}, indent=2))

        # Save cost information if provided
        if cost_data:
            self._save_cost_report(paper.id, cost_data)

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "publication_text.txt"),
            "w",
        ) as f:
            f.write(paper.publication_text)

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "si_text.txt"),
            "w",
        ) as f:
            f.write(paper.si_text)

    def _save_cost_report(self, paper_id: str, cost_data: dict[str, Any]):
        """Save cost information to JSON format."""

        # Save detailed cost report as JSON
        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper_id,
            "cost_breakdown_usd": cost_data.get("breakdown", {}),
            "total_cost_usd": cost_data.get("total_cost", 0.0),
            "model_info": cost_data.get("models", {}),
            "statistics": {
                "total_llm_calls": cost_data.get("total_calls", 0),
                "materials_processed": cost_data.get("materials_count", 0),
                "synthesis_extractions": cost_data.get("synthesis_calls", 0),
                "material_extractions": cost_data.get("material_calls", 0),
                "judge_evaluations": cost_data.get("judge_calls", 0),
            },
        }

        with self.fs.open(
            os.path.join(self.result_dir, paper_id, "cost_report.json"), "w"
        ) as f:
            f.write(json.dumps(cost_report, indent=2))

    def _ensure_dir(self, dir: str):
        if not self.fs.exists(dir):
            self.fs.makedirs(dir)
