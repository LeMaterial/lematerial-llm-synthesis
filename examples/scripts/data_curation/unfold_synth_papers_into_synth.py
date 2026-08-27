"""One-off script: (re)build LeMaterial/LeMat-Synth from
LeMaterial/LeMat-Synth-Papers.

LeMat-Synth-Papers "full" config has one row per paper, with
structured_synthesis and evaluations as JSON-encoded lists (one entry per
material found in that paper). LeMat-Synth is the "unfolded" form: one row
per material.

This script:
- Rebuilds LeMat-Synth's "full" config directly from LeMat-Synth-Papers'
  "full" config (arxiv/chemrxiv/omg24 splits), exploding each paper row into
  one row per material.
- Renames "default" -> "NeurIPS-AI4Mat-2025" (kept as-is content-wise, just
  renamed, matching what was done for LeMat-Synth-Papers).
- Adds a new config "high_score" containing only unfolded rows with
  evaluation.scores.overall_score > 4, still split by arxiv/chemrxiv/omg24.

Pushes to a "rebuild" branch; fast-forward main once confirmed clean, same
approach used for LeMat-Synth-Papers.
"""

import argparse
import json

from datasets import Dataset, DatasetDict, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import RepoCard
from huggingface_hub.repocard_data import DatasetCardData

SOURCE_REPO = "LeMaterial/LeMat-Synth-Papers"
DEST_REPO = "LeMaterial/LeMat-Synth"
BRANCH = "rebuild"

SPLITS = ["arxiv", "chemrxiv", "omg24"]

PAPER_FIELD_MAP = {
    "title": "paper_title",
    "published_date": "paper_published_date",
    "abstract": "paper_abstract",
    "doi": "paper_doi",
    "pdf_url": "paper_url",
}

PLACEHOLDER_NULL_FIELDS = [
    "images",
    "plot_data",
    "synthesis_extraction_performance_llm",
    "figure_extraction_performance_llm",
    "synthesis_extraction_performance_human",
    "figure_extraction_performance_human",
]


def unfold_paper_row(row: dict) -> list[dict]:
    recipes = (
        json.loads(row["structured_synthesis"])
        if row["structured_synthesis"]
        else []
    )
    evaluations = json.loads(row["evaluations"]) if row["evaluations"] else []
    evals_by_material = {
        e["material_name"]: e["evaluation"] for e in evaluations
    }

    paper_fields = {dest: row[src] for src, dest in PAPER_FIELD_MAP.items()}

    unfolded = []
    for entry in recipes:
        recipe = entry["recipe"]
        material_row = {
            "synthesized_material": recipe.get("target_compound"),
            "material_category": recipe.get("target_compound_type"),
            "synthesis_method": recipe.get("synthesis_method"),
            "structured_synthesis": recipe,
            "evaluation": evals_by_material.get(entry["material_name"]),
            **{f: None for f in PLACEHOLDER_NULL_FIELDS},
            **paper_fields,
        }
        unfolded.append(material_row)
    return unfolded


def unfold_split(split_name: str) -> Dataset:
    papers = load_dataset(SOURCE_REPO, "full", split=split_name)
    rows = []
    for row in papers:
        rows.extend(unfold_paper_row(row))
    print(f"{split_name}: {papers.num_rows} papers -> {len(rows)} materials")
    return Dataset.from_list(rows)


def build_full(dry_run: bool) -> dict[str, Dataset]:
    unfolded = {split: unfold_split(split) for split in SPLITS}

    # paper_doi is all-None in some splits (e.g. omg24), which datasets infers
    # as a null-typed column instead of string - force it back to string
    # everywhere so DatasetDict.push_to_hub doesn't reject mismatched
    # features across splits.
    unfolded = {
        split: ds.cast_column("paper_doi", Value("string"))
        for split, ds in unfolded.items()
    }

    if not dry_run:
        DatasetDict(unfolded).push_to_hub(
            DEST_REPO, config_name="full", revision=BRANCH
        )
    return unfolded


def build_high_score(unfolded: dict[str, Dataset], dry_run: bool) -> None:
    filtered = {}
    for split, ds in unfolded.items():
        subset = ds.filter(
            lambda ex: (
                ex["evaluation"] is not None
                and ex["evaluation"]["scores"]["overall_score"] > 4
            )
        )
        filtered[split] = subset
        print(
            f"high_score/{split}: {subset.num_rows} / {ds.num_rows} rows kept"
        )
    if not dry_run:
        DatasetDict(filtered).push_to_hub(
            DEST_REPO, config_name="high_score", revision=BRANCH
        )


def rename_default(dry_run: bool) -> None:
    default = load_dataset(DEST_REPO, "default")
    print("default splits:", {k: v.num_rows for k, v in default.items()})
    if not dry_run:
        default.push_to_hub(
            DEST_REPO, config_name="NeurIPS-AI4Mat-2025", revision=BRANCH
        )


def delete_old_files(api: HfApi, dry_run: bool) -> None:
    revision = BRANCH if not dry_run else "main"
    files = api.list_repo_files(
        DEST_REPO, repo_type="dataset", revision=revision
    )
    to_delete = [f for f in files if f.startswith("data/")]
    print(f"Deleting {len(to_delete)} old files:")
    for f in to_delete:
        print(f"  {f}")
    if not dry_run:
        for f in to_delete:
            api.delete_file(
                f, repo_id=DEST_REPO, repo_type="dataset", revision=BRANCH
            )


def fix_readme(api: HfApi, dry_run: bool) -> None:
    revision = BRANCH if not dry_run else "main"
    readme_path = hf_hub_download(
        DEST_REPO,
        "README.md",
        repo_type="dataset",
        revision=revision,
        force_download=True,
    )
    card = RepoCard.load(readme_path)
    meta = card.data.to_dict()

    meta["dataset_info"] = [
        c for c in meta["dataset_info"] if c["config_name"] != "default"
    ]
    meta["configs"] = [
        c for c in meta["configs"] if c["config_name"] != "default"
    ]

    print(
        "Remaining configs after fixup:",
        [c["config_name"] for c in meta["configs"]],
    )

    if not dry_run:
        card.data = DatasetCardData(**meta)
        card.save("/tmp/README_synth_fixed.md")
        api.upload_file(
            path_or_fileobj="/tmp/README_synth_fixed.md",
            path_in_repo="README.md",
            repo_id=DEST_REPO,
            repo_type="dataset",
            revision=BRANCH,
            commit_message=(
                "Fix README after rebuild + rename + high_score subset"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.dry_run:
        api.create_branch(
            DEST_REPO, repo_type="dataset", branch=BRANCH, exist_ok=True
        )

    unfolded = build_full(args.dry_run)
    build_high_score(unfolded, args.dry_run)
    rename_default(args.dry_run)
    delete_old_files(api, args.dry_run)
    fix_readme(api, args.dry_run)

    if not args.dry_run:
        print(
            f"\nDone. Branch '{BRANCH}' ready. Fast-forward main with:\n"
            f"  git push origin {BRANCH}:main"
        )
