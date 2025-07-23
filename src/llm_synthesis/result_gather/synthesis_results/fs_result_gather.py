import json
import os

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

    def gather(self, paper: PaperWithSynthesisOntologies):
        self._ensure_dir(os.path.join(self.result_dir, paper.id))

        # Save the complete synthesis results
        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "result.json"), "w"
        ) as f:
            if paper.all_syntheses:
                synthesis_results = []
                for synthesis_entry in paper.all_syntheses:
                    result_data = {
                        "material": synthesis_entry.material,
                        "synthesis": synthesis_entry.synthesis.model_dump()
                        if synthesis_entry.synthesis
                        else None,
                        "evaluation_score": (
                            synthesis_entry.evaluation.scores.overall_score
                            if synthesis_entry.evaluation
                            else None
                        ),
                        "success": synthesis_entry.synthesis is not None,
                    }
                    synthesis_results.append(result_data)
                f.write(json.dumps(synthesis_results, indent=2))
            else:
                f.write(json.dumps({"error": "No synthesis found"}, indent=2))

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

        # Save evaluation results separately
        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "evaluation.json"),
            "w",
        ) as f:
            evaluation_results = []
            for synthesis_entry in paper.all_syntheses:
                eval_data = {
                    "material": synthesis_entry.material,
                    "evaluation": synthesis_entry.evaluation.model_dump()
                    if synthesis_entry.evaluation
                    else None,
                    "synthesis_available": synthesis_entry.synthesis
                    is not None,
                }
                evaluation_results.append(eval_data)
            f.write(json.dumps(evaluation_results, indent=2))

        # Save a summary of results
        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "summary.json"),
            "w",
        ) as f:
            total_materials = len(paper.all_syntheses)
            successful_extractions = sum(
                1 for s in paper.all_syntheses if s.synthesis is not None
            )
            successful_evaluations = sum(
                1 for s in paper.all_syntheses if s.evaluation is not None
            )
            avg_score = None
            if successful_evaluations > 0:
                scores = [
                    s.evaluation.scores.overall_score
                    for s in paper.all_syntheses
                    if s.evaluation
                ]
                avg_score = sum(scores) / len(scores)

            summary = {
                "paper_id": paper.id,
                "paper_name": paper.name,
                "total_materials": total_materials,
                "successful_extractions": successful_extractions,
                "successful_evaluations": successful_evaluations,
                "extraction_success_rate": successful_extractions
                / total_materials
                if total_materials > 0
                else 0,
                "evaluation_success_rate": successful_evaluations
                / total_materials
                if total_materials > 0
                else 0,
                "average_evaluation_score": avg_score,
                "materials": [s.material for s in paper.all_syntheses],
            }
            f.write(json.dumps(summary, indent=2))

    def _ensure_dir(self, dir: str):
        if not self.fs.exists(dir):
            self.fs.makedirs(dir)
