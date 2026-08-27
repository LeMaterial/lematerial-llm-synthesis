"""One-off script: fold *_recipes_and_evals configs into the "full" config.

- full/arxiv    <- arxiv_recipes_and_evals/full    (replaces existing
  full/arxiv)
- full/chemrxiv <- chemrxiv_recipes_and_evals/full (replaces existing
  full/chemrxiv)
- full/omg24    <- omg24_recipes_and_evals/full    (replaces existing
  full/omg24)
- deletes configs arxiv_recipes_and_evals, chemrxiv_recipes_and_evals,
  omg24_recipes_and_evals

Pushes straight to a "merge-recipes" branch, then fast-forwards main once
confirmed clean (mirrors the approach used for the earlier prune, since
PR-from-existing-branch tooling on the Hub doesn't work smoothly). Run with
--dry-run first.
"""

import argparse

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.repocard import RepoCard
from huggingface_hub.repocard_data import DatasetCardData

REPO_ID = "LeMaterial/LeMat-Synth-Papers"
BRANCH = "merge-recipes"

# source_config -> dest split name in "full"
SOURCE_TO_SPLIT = {
    "arxiv_recipes_and_evals": "arxiv",
    "chemrxiv_recipes_and_evals": "chemrxiv",
    "omg24_recipes_and_evals": "omg24",
}


def build_new_full(dry_run: bool) -> None:
    new_splits = {}
    for source_config, split_name in SOURCE_TO_SPLIT.items():
        ds = load_dataset(REPO_ID, source_config, split="full")
        new_splits[split_name] = ds
        print(
            f"{source_config}/full -> full/{split_name}: "
            f"{ds.num_rows} rows, columns={ds.column_names}"
        )

    if not dry_run:
        DatasetDict(new_splits).push_to_hub(
            REPO_ID, config_name="full", revision=BRANCH
        )


def delete_old_configs(api: HfApi, dry_run: bool) -> None:
    revision = BRANCH if not dry_run else "main"
    files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
    to_delete = [
        f for f in files if any(f.startswith(f"{c}/") for c in SOURCE_TO_SPLIT)
    ]
    print(f"Deleting {len(to_delete)} old files:")
    for f in to_delete:
        print(f"  {f}")
    if not dry_run:
        for f in to_delete:
            api.delete_file(
                f, repo_id=REPO_ID, repo_type="dataset", revision=BRANCH
            )


def fix_readme(api: HfApi, dry_run: bool) -> None:
    revision = BRANCH if not dry_run else "main"
    readme_path = hf_hub_download(
        REPO_ID,
        "README.md",
        repo_type="dataset",
        revision=revision,
        force_download=True,
    )
    card = RepoCard.load(readme_path)
    meta = card.data.to_dict()

    di = [
        c
        for c in meta["dataset_info"]
        if c["config_name"] not in SOURCE_TO_SPLIT
    ]
    cfgs = [
        c for c in meta["configs"] if c["config_name"] not in SOURCE_TO_SPLIT
    ]
    meta["dataset_info"] = di
    meta["configs"] = cfgs

    print("Remaining configs after fixup:", [c["config_name"] for c in cfgs])

    if not dry_run:
        card.data = DatasetCardData(**meta)
        card.save("/tmp/README_merge_fixed.md")
        api.upload_file(
            path_or_fileobj="/tmp/README_merge_fixed.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=BRANCH,
            commit_message=(
                "Fix README after merging recipes_and_evals into full"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.dry_run:
        api.create_branch(
            REPO_ID, repo_type="dataset", branch=BRANCH, exist_ok=True
        )

    build_new_full(args.dry_run)
    delete_old_configs(api, args.dry_run)
    fix_readme(api, args.dry_run)

    if not args.dry_run:
        print(
            f"\nDone. Branch '{BRANCH}' ready. Fast-forward main with:\n"
            f"  git push origin {BRANCH}:main"
        )
