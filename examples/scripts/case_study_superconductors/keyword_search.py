"""
Keyword search for superconductor papers.

Pipeline:
1. Load all three splits (arxiv, omg24, chemrxiv)
2. Filter by category: "Superconductor" (substring match, case-insensitive)
3. Filter by keyword: "resistivity" in abstract
4. Export matched paper IDs to pickle file for downstream LLM filtering
"""

import pickle

from datasets import concatenate_datasets, load_dataset


# --- Category filter ---
def filter_by_category(ds, category_filter):
    """Keep papers whose categories string contains any of the filter terms."""
    if not category_filter:
        return ds
    cat_lower = [c.lower() for c in category_filter]

    def has_category(example):
        cats = example["categories"]
        if cats is None:
            return False
        cats_lower = cats.lower()
        return any(cf in cats_lower for cf in cat_lower)

    return ds.filter(has_category)


# --- Keyword filter ---
def keyword_filter(ds, text_column, include_kws, exclude_kws=None):
    """Filter dataset by include/exclude keywords (case-insensitive)."""

    def matches_include(example):
        if example[text_column] is None:
            return False
        text_lower = example[text_column].lower()
        return any(kw.lower() in text_lower for kw in include_kws)

    filtered = ds.filter(matches_include)

    if exclude_kws:

        def matches_exclude(example):
            if example[text_column] is None:
                return False
            text_lower = example[text_column].lower()
            return any(ek.lower() in text_lower for ek in exclude_kws)

        filtered = filtered.filter(lambda x: not matches_exclude(x))

    return filtered


if __name__ == "__main__":
    # Load the "full" configuration
    dataset = load_dataset(
        "LeMaterial/LeMat-Synth-Papers",
        "full",
        split=None,
        token=True,
    )
    print("Loaded dataset with columns:", dataset.column_names)

    # Use all three splits
    split_name_list = ["arxiv", "omg24", "chemrxiv"]
    category_filter = ["Superconductor"]
    text_column = "abstract"
    include_keywords = ["resistivity"]

    db = {}

    for split_name in split_name_list:
        ds = dataset[split_name]
        print(f"\n{'=' * 60}")
        print(f"Processing split: {split_name} ({len(ds)} papers)")
        print(f"{'=' * 60}")

        # Step 1: Filter by category
        ds_cat = filter_by_category(ds, category_filter)
        print(
            f"  Category filter {category_filter}: "
            f"{len(ds_cat)} / {len(ds)} papers"
        )

        # Step 2: Filter by keyword "resistivity" in abstract
        ds_filtered = keyword_filter(ds_cat, text_column, include_keywords)
        print(
            f"  Keyword filter {include_keywords}: "
            f"{len(ds_filtered)} / {len(ds_cat)} papers"
        )

        # Collect IDs
        ids = set(ds_filtered["id"])
        db[split_name] = ids
        print(f"  Final IDs for {split_name}: {len(ids)}")

    # Summary
    all_ids = set()
    for split_name, ids in db.items():
        all_ids.update(ids)
    print(f"\n{'=' * 60}")
    print(f"TOTAL unique papers across all splits: {len(all_ids)}")
    print(f"{'=' * 60}")

    # Export to pickle
    output_path = "results/db_superconductors.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(db, f)

    print(f"\nSaved {output_path} with keys: {list(db.keys())}")
