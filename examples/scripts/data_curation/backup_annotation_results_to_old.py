#!/usr/bin/env python3
"""
Within each paper folder under annotations/:
1. Create an "old" subfolder and copy result.json and result_human.json into it.
2. Write a new result_human.json in the folder root in multi-LLM format
   (schema_version multi_llm_v1) per docs/multi_llm_annotation_design.md.
"""

import argparse
import json
import shutil
from pathlib import Path

FILES_TO_BACKUP = ("result.json", "result_human.json")
SCHEMA_VERSION = "multi_llm_v1"
EXTRACTOR_ORDER = [
    "claude-sonnet-4.6",
    "gemini-3-flash",
    "qwen3.5-397b-a17b",
    "deepseek-v3.2",
]

EMPTY_HUMAN_RECIPE = {
    "target_compound": None,
    "target_compound_type": None,
    "synthesis_method": None,
    "starting_materials": [],
    "steps": [],
    "equipment": [],
    "notes": None,
}

EMPTY_EVALUATION = {
    "evaluation": {
        "reasoning": "",
        "scores": {
            "structural_completeness_score": None,
            "structural_completeness_reasoning": "",
            "material_extraction_score": None,
            "material_extraction_reasoning": "",
            "process_steps_score": None,
            "process_steps_reasoning": "",
            "equipment_extraction_score": None,
            "equipment_extraction_reasoning": "",
            "conditions_extraction_score": None,
            "conditions_extraction_reasoning": "",
            "semantic_accuracy_score": None,
            "semantic_accuracy_reasoning": "",
            "format_compliance_score": None,
            "format_compliance_reasoning": "",
            "overall_score": None,
            "overall_reasoning": "",
        },
        "confidence_level": None,
        "missing_information": [],
        "extraction_errors": [],
        "improvement_suggestions": [],
    }
}


def _empty_evaluation_block():
    """Return a deep copy of the empty evaluation block (4 slots)."""
    return [json.loads(json.dumps(EMPTY_EVALUATION)) for _ in range(4)]


def _empty_human_recipe():
    return json.loads(json.dumps(EMPTY_HUMAN_RECIPE))


def _material_name_from_old_human(entry: dict) -> str:
    return entry.get("material") or entry.get("material_name") or ""


def load_paper_id_to_url(papers_path: Path | None) -> dict[str, str]:
    """Load paper id -> url from papers JSON (id, pdf_url)."""
    if not papers_path or not papers_path.is_file():
        return {}
    try:
        with open(papers_path, encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("id")
        if pid is None:
            continue
        url = row.get("pdf_url") or row.get("paper_url") or ""
        out[str(pid)] = "" if url is None else str(url)
    return out


def build_multi_llm_human(
    paper_id: str,
    result_json_entries: list[dict],
    old_human_entries: list[dict] | None,
    paper_url: str = "",
) -> dict:
    """Build multi_llm_v1 result_human.json structure."""
    old_by_material = {}
    if old_human_entries:
        for e in old_human_entries:
            name = _material_name_from_old_human(e)
            if name:
                old_by_material[name] = e

    materials = []
    for entry in result_json_entries:
        material_name = entry.get("material") or ""
        if not material_name:
            continue
        old_entry = old_by_material.get(material_name)
        if old_entry and "synthesis" in old_entry:
            human_recipe = json.loads(json.dumps(old_entry["synthesis"]))
        else:
            human_recipe = _empty_human_recipe()
        materials.append(
            {
                "material_name": material_name,
                "human_recipe": human_recipe,
                "evaluations": _empty_evaluation_block(),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "paper_url": paper_url or "",
        "extractor_order": EXTRACTOR_ORDER,
        "materials": materials,
    }


def process_paper_folder(
    paper_dir: Path,
    paper_id_to_url: dict[str, str],
) -> tuple[int, bool]:
    """
    Backup result.json and result_human.json to old/, then write new
    result_human.json. Returns (files_copied, new_file_written).
    """
    old_dir = paper_dir / "old"
    copied = 0
    for name in FILES_TO_BACKUP:
        src = paper_dir / name
        if src.is_file():
            old_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, old_dir / name)
            copied += 1

    result_path = paper_dir / "result.json"
    if not result_path.is_file():
        return copied, False

    with open(result_path, encoding="utf-8") as f:
        result_entries = json.load(f)
    if not isinstance(result_entries, list) or not result_entries:
        return copied, False

    old_human_path = paper_dir / "result_human.json"
    old_human_entries = None
    if old_human_path.is_file():
        try:
            with open(old_human_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                old_human_entries = data
        except (json.JSONDecodeError, OSError):
            pass

    paper_url = paper_id_to_url.get(paper_dir.name, "")
    payload = build_multi_llm_human(
        paper_dir.name,
        result_entries,
        old_human_entries,
        paper_url=paper_url,
    )
    out_path = paper_dir / "result_human.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return copied, True


def main():
    parser = argparse.ArgumentParser(
        description="Backup to old/ and write multi-LLM result_human.json"
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("annotations"),
        help="Root annotations dir (default: annotations)",
    )
    default_papers = Path("examples/data/annotation_papers.json")
    parser.add_argument(
        "--papers-json",
        type=Path,
        default=default_papers,
        help=f"JSON with paper rows (id, pdf_url). Default: {default_papers}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, do not write.",
    )
    args = parser.parse_args()

    root = args.annotations_dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    paper_id_to_url = load_paper_id_to_url(args.papers_json.resolve())
    paper_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    total_copied = 0
    total_new = 0
    for paper_dir in paper_dirs:
        if args.dry_run:
            to_copy = [f for f in FILES_TO_BACKUP if (paper_dir / f).is_file()]
            has_result = (paper_dir / "result.json").is_file()
            if to_copy or has_result:
                s = (
                    f"{paper_dir.name}: copy {to_copy} -> old/, "
                    f"new={has_result}"
                )
                print(s)
                total_copied += len(to_copy)
                if has_result:
                    total_new += 1
        else:
            copied, new_written = process_paper_folder(
                paper_dir, paper_id_to_url
            )
            if copied or new_written:
                parts = []
                if copied:
                    parts.append(f"{copied} file(s) -> old/")
                if new_written:
                    parts.append("result_human.json (multi_llm_v1) written")
                print(f"{paper_dir.name}: {', '.join(parts)}")
                total_copied += copied
                if new_written:
                    total_new += 1

    print(f"Done. In old/: {total_copied}. New result_human.json: {total_new}")


if __name__ == "__main__":
    main()
