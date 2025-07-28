import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

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

    def gather(self, paper: PaperWithSynthesisOntologies, cost_data: Optional[Dict[str, Any]] = None):
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

    def _save_cost_report(self, paper_id: str, cost_data: Dict[str, Any]):
        """Save cost information to both JSON and human-readable text formats."""
        
        # Save detailed cost report as JSON
        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper_id,
            "cost_breakdown": cost_data.get("breakdown", {}),
            "total_cost_usd": cost_data.get("total_cost", 0.0),
            "summary": cost_data.get("summary", ""),
            "model_info": cost_data.get("models", {}),
            "statistics": {
                "total_llm_calls": cost_data.get("total_calls", 0),
                "materials_processed": cost_data.get("materials_count", 0),
                "synthesis_extractions": cost_data.get("synthesis_calls", 0),
                "material_extractions": cost_data.get("material_calls", 0),
                "judge_evaluations": cost_data.get("judge_calls", 0)
            }
        }
        
        with self.fs.open(
            os.path.join(self.result_dir, paper_id, "cost_report.json"), "w"
        ) as f:
            f.write(json.dumps(cost_report, indent=2))
        
        # Save human-readable cost summary as text
        cost_text = self._format_cost_text_report(cost_report)
        with self.fs.open(
            os.path.join(self.result_dir, paper_id, "cost_summary.txt"), "w"
        ) as f:
            f.write(cost_text)

    def _format_cost_text_report(self, cost_report: Dict[str, Any]) -> str:
        """Format cost report as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("LLM COST REPORT")
        lines.append("=" * 60)
        lines.append(f"Paper ID: {cost_report['paper_id']}")
        lines.append(f"Timestamp: {cost_report['timestamp']}")
        lines.append("")
        
        # Total cost
        total_cost = cost_report.get('total_cost_usd', 0.0)
        lines.append(f"💰 TOTAL COST: ${total_cost:.6f}")
        lines.append("")
        
        # Cost breakdown
        breakdown = cost_report.get('cost_breakdown', {})
        if breakdown:
            lines.append("📊 COST BREAKDOWN BY COMPONENT:")
            for component, cost in breakdown.items():
                lines.append(f"   • {component.replace('_', ' ').title()}: ${cost:.6f}")
            lines.append("")
        
        # Statistics
        stats = cost_report.get('statistics', {})
        if stats:
            lines.append("📈 USAGE STATISTICS:")
            lines.append(f"   • Total LLM calls: {stats.get('total_llm_calls', 0)}")
            lines.append(f"   • Materials processed: {stats.get('materials_processed', 0)}")
            lines.append(f"   • Synthesis extractions: {stats.get('synthesis_extractions', 0)}")
            lines.append(f"   • Material extractions: {stats.get('material_extractions', 0)}")
            lines.append(f"   • Judge evaluations: {stats.get('judge_evaluations', 0)}")
            lines.append("")
        
        # Model information
        models = cost_report.get('model_info', {})
        if models:
            lines.append("🤖 MODELS USED:")
            for component, model in models.items():
                lines.append(f"   • {component.replace('_', ' ').title()}: {model}")
            lines.append("")
        
        # Summary
        summary = cost_report.get('summary', '')
        if summary:
            lines.append("📝 SUMMARY:")
            lines.append(f"   {summary}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def _ensure_dir(self, dir: str):
        if not self.fs.exists(dir):
            self.fs.makedirs(dir)
