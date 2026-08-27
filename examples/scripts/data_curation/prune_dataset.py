"""One-off script to prune LeMaterial/LeMat-Synth-Papers.

- Deletes configs: superconductor_keywords_only,
  thermocatalysis_keyword_LLM_strict,
  thermocatalysis_keyword_LLM_strict_for_cat,
  thermocatalysis_keyword_LLM_strict_for_conversion
- Drops splits thermocatalysis_keywords_only / thermocatalysis_keywords_and_LLM
  from "full"
- Renames "default" (split sample_for_evaluation) to "NeurIPS-AI4Mat-2025"
- Drops columns updated_date, views_count, read_count, citation_count, keywords,
  pdf_extractor, images from every remaining config

Re-uploads each surviving config/split as its own DatasetDict push, which
rewrites the README.md dataset_info/configs YAML to match automatically.

Run with --dry-run first to see what would happen without pushing anything.
"""

import argparse

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

REPO_ID = "LeMaterial/LeMat-Synth-Papers"
PR_BRANCH = "prune-dataset"

DROP_COLUMNS = [
    "updated_date",
    "views_count",
    "read_count",
    "citation_count",
    "keywords",
    "pdf_extractor",
    "images",
]

CONFIGS_TO_DELETE = [
    "superconductor_keywords_only",
    "thermocatalysis_keyword_LLM_strict",
    "thermocatalysis_keyword_LLM_strict_for_cat",
    "thermocatalysis_keyword_LLM_strict_for_conversion",
]

SPLITS_TO_DROP_FROM_FULL = [
    "thermocatalysis_keywords_only",
    "thermocatalysis_keywords_and_LLM",
]

# (source_config, source_split) -> (dest_config, dest_split)
RENAMES = {
    ("default", "sample_for_evaluation"): (
        "NeurIPS-AI4Mat-2025",
        "sample_for_evaluation",
    ),
}

CONFIGS_TO_KEEP = [
    "arxiv_recipes_and_evals",
    "chemrxiv_recipes_and_evals",
    "default",
    "full",
    "omg24_recipes_and_evals",
    "superconductor_keywords_and_LLM",
]


def drop_columns(ds: Dataset) -> Dataset:
    cols = [c for c in DROP_COLUMNS if c in ds.column_names]
    return ds.remove_columns(cols) if cols else ds


def process_config(config: str, dry_run: bool) -> None:
    splits = load_dataset(REPO_ID, config)
    assert isinstance(splits, DatasetDict)

    if config == "full":
        for split in SPLITS_TO_DROP_FROM_FULL:
            splits.pop(split, None)

    dest_config = config
    new_dict = {}
    for split_name, ds in splits.items():
        dest_config, dest_split = RENAMES.get(
            (config, split_name), (config, split_name)
        )
        new_dict[dest_split] = drop_columns(ds)

    print(
        f"{config} -> {dest_config}: splits={list(new_dict.keys())}, "
        f"columns={list(next(iter(new_dict.values())).column_names)}"
    )

    if not dry_run:
        DatasetDict(new_dict).push_to_hub(
            REPO_ID, config_name=dest_config, revision=PR_BRANCH
        )


def delete_old_config_files(api: HfApi, dry_run: bool) -> None:
    # NB: push_to_hub already removes the old shard files for any config it
    # rewrites (including dropped "full" splits and the
    # default->NeurIPS-AI4Mat-2025 rename). Only the fully-deleted configs'
    # directories still need cleanup.
    files = api.list_repo_files(
        REPO_ID, repo_type="dataset", revision=PR_BRANCH
    )

    to_delete = [
        f
        for f in files
        if any(f.startswith(f"{c}/") for c in CONFIGS_TO_DELETE)
    ]

    print(f"Deleting {len(to_delete)} old files:")
    for f in to_delete:
        print(f"  {f}")

    if not dry_run:
        for f in to_delete:
            api.delete_file(
                f, repo_id=REPO_ID, repo_type="dataset", revision=PR_BRANCH
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.dry_run:
        api.create_branch(
            REPO_ID, repo_type="dataset", branch=PR_BRANCH, exist_ok=True
        )

    for config in CONFIGS_TO_KEEP:
        process_config(config, args.dry_run)

    delete_old_config_files(api, args.dry_run)

    if not args.dry_run:
        # huggingface_hub has no API to convert an existing branch into a PR;
        # open it manually from the branch-compare view.
        print(
            "\nAll changes committed to branch "
            f"'{PR_BRANCH}'. Open a PR here:\n"
            f"https://huggingface.co/datasets/{REPO_ID}/compare/main...{PR_BRANCH}"
        )
