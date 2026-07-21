"""Publish the omg24 qwen recipes as the LeMat-Synth-Papers omg24 config.

Follows the same "joined" pattern used for ``superconductor_keywords_and_LLM``:
keep EVERY source omg24 paper row and column, fill ``structured_synthesis``
(JSON string), ``evaluations`` (JSON string) and the 7 rubric scores +
``overall_score`` where a recipe was extracted, and leave them NULL where it was
not. The cleaned recipe/eval/score columns come from the private v2 results repo
and are joined onto the source omg24 paper rows by id.

Output lands in the ``omg24_recipes_and_evals`` config (its own single-split
config, exactly like superconductor) — NOT inside ``full``, whose five splits
share one schema and would break if omg24's changed.

Run on the cluster (the source omg24 split carries full paper text + images, so
it is a multi-GB download; it caches to disk so a verify run and the ``--push``
run download only once). Alignment uses only the id column, so paper text/images
are never decoded into Python.

.. code-block:: bash

    # verify: build the join + print counts, no upload
    uv run --with datasets python \\
        examples/scripts/deployment/publish_omg24_recipes_config.py

    # build + push the config
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
    so their source rows join to NULL (no recipe extracted).

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
        help="upload the config (omit to only build + print a summary)",
    )
    args = ap.parse_args()

    recipes = build_recipe_map()
    print(f"Recipe rows in v2 (with a recipe): {len(recipes)}")

    # Full (cached) load so image/text columns keep their native Arrow types.
    src = load_dataset(TARGET_REPO, SOURCE_CONFIG, split=SOURCE_SPLIT)
    n = src.num_rows
    print(f"Source omg24 rows: {n}")

    # Build aligned columns by id only (never decodes text/images), NULL where
    # no recipe was extracted.
    new_ss: list[str | None] = []
    new_eval: list[str | None] = []
    new_scores: dict[str, list[float | None]] = {c: [] for c in SCORE_COLUMNS}
    matched = 0
    for rid in src["id"]:
        rec = recipes.get(rid)
        if rec is None:
            new_ss.append(None)
            new_eval.append(None)
            for c in SCORE_COLUMNS:
                new_scores[c].append(None)
            continue
        matched += 1
        new_ss.append(rec["structured_synthesis"])
        new_eval.append(rec["evaluations"])
        for c in SCORE_COLUMNS:
            v = rec[c]
            new_scores[c].append(v if isinstance(v, (int, float)) else None)
    print(f"Matched {matched}/{n} source rows to a recipe")

    # Replace the source struct structured_synthesis with the JSON string and
    # append the eval + score columns (NULL where unmatched).
    out = src.remove_columns("structured_synthesis")
    out = out.add_column("structured_synthesis", new_ss)
    out = out.add_column("evaluations", new_eval)
    for c in SCORE_COLUMNS:
        out = out.add_column(c, new_scores[c])
    # Pin scores to float64 so an all-null column can't infer to null type.
    for c in SCORE_COLUMNS:
        if out.features[c] != Value("float64"):
            out = out.cast_column(c, Value("float64"))

    # --- verification ---
    print("\n=== VERIFICATION ===")
    print(f"rows: {out.num_rows} (source had {n})")
    assert out.num_rows == n, "row count changed!"
    added = [c for c in out.column_names if c not in src.column_names]
    print(f"columns: {len(out.column_names)} | new: {added}")
    print("structured_synthesis feature:", out.features["structured_synthesis"])
    filled = sum(1 for v in out["structured_synthesis"] if v)
    scored = sum(
        1 for v in out["overall_score"] if isinstance(v, (int, float))
    )
    print(f"structured_synthesis filled: {filled}/{n}")
    print(f"overall_score filled: {scored}/{n}")

    if not args.push:
        print(
            f"\nBUILD OK — not pushed. Re-run with --push to publish "
            f"{TARGET_REPO}:{NEW_CONFIG} (cached, no re-download)."
        )
        return

    print(
        f"\nPushing {out.num_rows} rows to {TARGET_REPO} "
        f"(config={NEW_CONFIG}, split={NEW_SPLIT}) ..."
    )
    out.push_to_hub(TARGET_REPO, config_name=NEW_CONFIG, split=NEW_SPLIT)
    print("Pushed. omg24_recipes_and_evals now holds the full joined split.")


if __name__ == "__main__":
    main()
