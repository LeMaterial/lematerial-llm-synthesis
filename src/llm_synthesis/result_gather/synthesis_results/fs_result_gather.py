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


class MultiLLMResultGather(SynthesisFSResultGather):
    """Result gatherer for multi-LLM extraction pipelines.

    Saves result.json grouped by synthesis LLM, with each material's synthesis
    followed by evaluations from all judge LLMs.
    """

    def gather(
        self,
        paper_id: str,
        publication_text: str,
        si_text: str,
        multi_llm_results: list[dict],
        cost_data: list[dict] | None = None,
    ):
        """Save multi-LLM results for a paper.

        Args:
            paper_id: Unique paper identifier (used as directory name).
            publication_text: Full paper text.
            si_text: Supplementary information text.
            multi_llm_results: List of per-synth-LLM result dicts in the 
            following format

                [
                    {
                        "synth_llm": "model-name",
                        "materials": [
                            {
                                "material": "compound",
                                "synthesis": { ... },
                                "evaluations": [
                                    {
                                        "judge_llm": "model-name",
                                        "evaluation": { ... } or None,
                                        "overall_score": float or None
                                    },
                                    ...
                                ]
                            },
                            ...
                        ]
                    },
                    ...
                ]

            cost_data: Optional list of per-operation cost dicts.
        """
        paper_dir = os.path.join(self.result_dir, paper_id)
        self._ensure_dir(paper_dir)

        # Save result.json
        with self.fs.open(
            os.path.join(paper_dir, "result.json"), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(multi_llm_results, indent=2, ensure_ascii=False))

        # Save publication_text.txt
        with self.fs.open(
            os.path.join(paper_dir, "publication_text.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(publication_text)

        # Save si_text.txt
        with self.fs.open(
            os.path.join(paper_dir, "si_text.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(si_text)

        # Save cost report
        if cost_data:
            self._save_cost_report(paper_id, cost_data)

    def _save_cost_report(self, paper_id: str, cost_data: list[dict]):
        """Save cost report structured by synth_llm, with judge evaluation 
        costs nested.

        Format:
            [
                {
                    "synth_llm": "...",
                    "extraction_cost_usd": <material + synthesis total>,
                    "judges": [
                        { "judge_llm": "...", "evaluation_cost_usd": <total> },
                        ...
                    ]
                },
                ...
            ]
        """
        from collections import defaultdict

        extraction_cost: dict[str, float] = defaultdict(float)
        eval_cost: dict[tuple[str, str], float] = defaultdict(float)
        synth_order: list[str] = []
        judge_order: dict[str, list[str]] = defaultdict(list)
        other_cost = 0.0

        for op in cost_data:
            operation = op.get("operation", "")
            cost = op.get("cost_usd", 0.0)
            s_llm = op.get("synth_llm", "")
            j_llm = op.get("judge_llm", "")

            if operation in ("material_extraction", "synthesis_extraction"):
                if s_llm not in synth_order:
                    synth_order.append(s_llm)
                extraction_cost[s_llm] += cost
            elif operation == "evaluation":
                if s_llm not in synth_order:
                    synth_order.append(s_llm)
                if j_llm not in judge_order[s_llm]:
                    judge_order[s_llm].append(j_llm)
                eval_cost[(s_llm, j_llm)] += cost
            else:
                other_cost += cost

        breakdown = []
        for s_llm in synth_order:
            judges = [
                {
                    "judge_llm": j_llm,
                    "evaluation_cost_usd": eval_cost.get((s_llm, j_llm), 0.0),
                }
                for j_llm in judge_order.get(s_llm, [])
            ]
            breakdown.append(
                {
                    "synth_llm": s_llm,
                    "extraction_cost_usd": extraction_cost.get(s_llm, 0.0),
                    "judges": judges,
                }
            )

        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper_id,
            "breakdown": breakdown,
            "other_cost_usd": other_cost,
            "total_cost_usd": sum(op.get("cost_usd", 0.0) for op in cost_data),
        }
        with self.fs.open(
            os.path.join(self.result_dir, paper_id, "cost_report.json"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(cost_report, indent=2))
