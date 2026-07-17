"""Publish the omg24 qwen recipes as a new LeMat-Synth-Papers config.

Mirrors the ``superconductor_keywords_and_LLM`` config: every source omg24 paper
column, but with ``structured_synthesis`` as a JSON string plus ``evaluations``
(JSON string) and the seven rubric scores + ``overall_score`` (float). Only rows
that produced a recipe are included. The cleaned recipe/eval/score columns come
from the private v2 results repo and are joined onto the source omg24 paper rows
by id.

Run on the cluster (the source omg24 split carries full paper text + images, so
it is a multi-GB download). It caches to disk, so a verify run (no ``--push``)
and the real ``--push`` run download only once.

.. code-block:: bash

    # verify: build + print schema/counts, no upload
    uv run --with datasets python \\
        examples/scripts/deployment/publish_omg24_recipes_config.py

    # build + push the new config
    uv run --with datasets python \\
        examples/scripts/deployment/publish_omg24_recipes_config.py --push
"""

from __future__ import annotations

import argparse
import json

from datasets import Value, load_dataset

V2_REPO = "sid-betalol/LeMat-Synth-recipes-qwen-omg24-v2"
V2_CONFIG, V2_SPLIT = "full", "omg24"
TARGET_REPO = "LeMaterial/LeMat-Synth-Papers"
SOURCE_CONFIG, SOURCE_SPLIT = "full", "omg24"
NEW_CONFIG, NEW_SPLIT = "omg24_recipes_and_evals", "full"

SCORE_COLUMNS = [
    "structural_completeness_score",
    "material_extraction_score",
    "process_steps_score",
    "equipment_extraction_score",
    "conditions_extraction_score",
    "semantic_accuracy_score",
    "format_compliance_score",
    "overall_score",
]
RECIPE_COLS = ["structured_synthesis", "evaluations", *SCORE_COLUMNS]


def build_recipe_map() -> dict[str, dict]:
    """Return ``{id: recipe/eval/score columns}`` for v2 rows with a recipe.

    Rows with an error, or whose ``structured_synthesis`` is empty, are skipped
    so only papers that actually produced a recipe are published.

    Returns
    -------
    dict[str, dict]
        Mapping of paper id to its ``RECIPE_COLS`` values.
    """
    ds = load_dataset(V2_REPO, V2_CONFIG, split=V2_SPLIT)
    out: dict[str, dict] = {}
    for r in ds:
        if r.get("error") is not None:
            continue
        ss = r.get("structured_synthesis")
        try:
            recipes = json.loads(ss) if isinstance(ss, str) else (ss or [])
        except (json.JSONDecodeError, TypeError):
            recipes = []
        if not recipes:
            continue
        out[r["id"]] = {c: r.get(c) for c in RECIPE_COLS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--push",
        action="store_true",
        help="upload the new config (omit to only build + print a summary)",
    )
    args = ap.parse_args()

    recipes = build_recipe_map()
    ids = set(recipes)
    print(f"Recipe rows in v2 (with a recipe): {len(ids)}")

    # Full (cached) load so image/text columns keep their native Arrow types.
    src = load_dataset(TARGET_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)
    print(f"Source omg24 rows: {len(src)}")

    # Keep only the recipe-bearing papers, drop the source struct
    # structured_synthesis (we replace it with the JSON-string version).
    ds = src.filter(lambda r: r["id"] in ids, desc="filter recipe rows")
    ds = ds.remove_columns("structured_synthesis")

    def add_recipe(row: dict) -> dict:
        rec = recipes[row["id"]]
        return {
            "structured_synthesis": rec["structured_synthesis"],
            "evaluations": rec["evaluations"],
            **{c: rec[c] for c in SCORE_COLUMNS},
        }

    ds = ds.map(add_recipe, desc="attach recipes/evals/scores")
    # Pin score columns to float64 so an all-null batch can't infer them away.
    for c in SCORE_COLUMNS:
        if ds.features[c] != Value("float64"):
            ds = ds.cast_column(c, Value("float64"))

    print(f"Joined rows: {len(ds)}")
    print(f"Columns ({len(ds.column_names)}): {ds.column_names}")
    print("structured_synthesis feature:", ds.features["structured_synthesis"])
    missing = ids - set(ds["id"])
    print(f"Recipe ids not found in source omg24: {len(missing)}")
    s = ds[0]
    print(
        f"Sample row -> id={s['id']} overall_score={s['overall_score']} "
        f"n_recipes={len(json.loads(s['structured_synthesis']))}"
    )

    if not args.push:
        print(
            f"\nBUILD OK — not pushed. Re-run with --push to publish "
            f"{TARGET_REPO}:{NEW_CONFIG} (cached, so no re-download)."
        )
        return

    print(
        f"\nPushing {len(ds)} rows to {TARGET_REPO} "
        f"(config={NEW_CONFIG}, split={NEW_SPLIT}) ..."
    )
    ds.push_to_hub(TARGET_REPO, config_name=NEW_CONFIG, split=NEW_SPLIT)
    print("Pushed. New config added; existing configs untouched.")


if __name__ == "__main__":
    main()
