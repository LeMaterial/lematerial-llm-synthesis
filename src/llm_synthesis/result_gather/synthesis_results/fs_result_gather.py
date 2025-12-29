import json
import os
from datetime import datetime

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

        if paper.cost_data:
            self._save_cost_report(paper)

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

    def _save_cost_report(self, paper: PaperWithSynthesisOntologies):
        """Save cost information to JSON format."""

        # Save detailed cost report as JSON
        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper.id,
            "cost_breakdown_usd": paper.cost_data.get("breakdown", {}),
            "total_cost_usd": paper.cost_data.get("total_cost", 0.0),
            "model_info": paper.cost_data.get("models", {}),
            "statistics": {
                "total_llm_calls": paper.cost_data.get("total_calls", 0),
                "materials_processed": paper.cost_data.get(
                    "materials_count", 0
                ),
                "synthesis_extractions": paper.cost_data.get(
                    "synthesis_calls", 0
                ),
                "material_extractions": paper.cost_data.get(
                    "material_calls", 0
                ),
                "judge_evaluations": paper.cost_data.get("judge_calls", 0),
            },
        }

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "cost_report.json"), "w"
        ) as f:
            f.write(json.dumps(cost_report, indent=2))

    def _ensure_dir(self, dir: str):
        if not self.fs.exists(dir):
            self.fs.makedirs(dir)
