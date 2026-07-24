"""Join local extraction results onto a LeMat-Synth-Papers split and publish.

Generalized version of the omg24 publish: reads the results JSONL written by
``run_extract_judge_hf.py`` (LOCAL file, no personal HF repo), joins it onto the
source ``full/<split>`` paper rows by id, and pushes a new
``<split>_recipes_and_evals`` config to LeMaterial with the same schema as
``omg24_recipes_and_evals``:

  * every source paper column kept as-is,
  * ``structured_synthesis`` replaced (source struct -> JSON string list),
  * ``evaluations`` (JSON string) + the 7 rubric scores + ``overall_score``
    (float) added,
  * NULL for papers with no extracted recipe (joined, all rows kept).

Runs on the cluster (source split is a multi-GB download; cached, so a verify
run and the ``--push`` run download only once). Needs an HF token with WRITE on
LeMaterial. Alignment uses only the id column, so text/images are never decoded.

.. code-block:: bash

    # verify (no upload)
    uv run --with datasets python \\
        examples/scripts/deployment/publish_recipes_config.py --split arxiv

    # build + push
    uv run --with datasets python \\
        examples/scripts/deployment/publish_recipes_config.py \\
        --split arxiv --push
"""

from __future__ import annotations

import argparse
import json

from datasets import Value, load_dataset

TARGET_REPO = "LeMaterial/LeMat-Synth-Papers"
SOURCE_CONFIG = "full"
# every *_recipes_and_evals config uses a single "full" split
NEW_SPLIT = "full"

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


def build_recipe_map(results_path: str) -> dict[str, dict]:
    """Return ``{id: recipe/eval/score columns}`` for rows with a recipe.

    Reads the crash-safe JSONL from ``run_extract_judge_hf.py``. An id can occur
    more than once (a retry appended after an error line); the last occurrence
    wins. Rows with an error or an empty recipe are skipped so their source rows
    join to NULL.

    Parameters
    ----------
    results_path : str
        Path to the results JSONL.

    Returns
    -------
    dict[str, dict]
        Mapping of paper id to its ``RECIPE_COLS`` values.
    """
    by_id: dict[str, dict] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id") is not None:
                by_id[r["id"]] = r
    out: dict[str, dict] = {}
    for rid, r in by_id.items():
        if r.get("error") is not None:
            continue
        ss = r.get("structured_synthesis")
        try:
            recipes = json.loads(ss) if isinstance(ss, str) else (ss or [])
        except (json.JSONDecodeError, TypeError):
            recipes = []
        if not recipes:
            continue
        out[rid] = {c: r.get(c) for c in RECIPE_COLS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split", required=True, help="source split, e.g. arxiv / chemrxiv"
    )
    ap.add_argument(
        "--results",
        default=None,
        help="results JSONL (default: results/full_<split>.jsonl)",
    )
    ap.add_argument(
        "--target-config",
        default=None,
        help="target config name (default: <split>_recipes_and_evals)",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="upload the config (omit to only build + print a summary)",
    )
    args = ap.parse_args()
    results = args.results or f"results/full_{args.split}.jsonl"
    target_config = args.target_config or f"{args.split}_recipes_and_evals"

    recipes = build_recipe_map(results)
    print(f"Recipe rows in {results} (with a recipe): {len(recipes)}")

    # Full (cached) load so image/text columns keep their native Arrow types.
    src = load_dataset(TARGET_REPO, SOURCE_CONFIG, split=args.split)
    n = src.num_rows
    print(f"Source {SOURCE_CONFIG}/{args.split} rows: {n}")

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

    out = src.remove_columns("structured_synthesis")
    out = out.add_column("structured_synthesis", new_ss)
    out = out.add_column("evaluations", new_eval)
    for c in SCORE_COLUMNS:
        out = out.add_column(c, new_scores[c])
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
    scored = sum(1 for v in out["overall_score"] if isinstance(v, (int, float)))
    print(f"structured_synthesis filled: {filled}/{n}")
    print(f"overall_score filled: {scored}/{n}")

    if not args.push:
        print(
            f"\nBUILD OK — not pushed. Re-run with --push to publish "
            f"{TARGET_REPO}:{target_config} (cached, no re-download)."
        )
        return

    print(
        f"\nPushing {out.num_rows} rows to {TARGET_REPO} "
        f"(config={target_config}, split={NEW_SPLIT}) ..."
    )
    out.push_to_hub(TARGET_REPO, config_name=target_config, split=NEW_SPLIT)
    print(f"Pushed. {target_config} holds the full joined split.")


if __name__ == "__main__":
    main()
