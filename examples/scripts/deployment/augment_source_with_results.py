"""Augment a LeMat-Synth-Papers split with extraction + judge results.

Non-destructive: loads the FULL source split (all original columns), joins the
results by ``id``, and produces a new split that

  * keeps every existing column unchanged,
  * fills ``structured_synthesis`` with the extracted recipe list as a JSON
    string (the source column is a single-struct schema that cannot hold our
    richer multi-material ontology, so it is replaced by a JSON string), and
  * adds new columns: ``evaluations`` (JSON string) and the 8 flat score
    columns.

By default it only VERIFIES (no push). Pass --target-repo to push, and start
with a COPY repo, not the canonical dataset.

Usage:
    # verify only (no push)
    uv run --with datasets python \
        examples/scripts/deployment/augment_source_with_results.py

    # push the augmented split to a COPY under LeMaterial, then inspect it
    uv run --with datasets python \
        examples/scripts/deployment/augment_source_with_results.py \
        --target-repo LeMaterial/LeMat-Synth-Papers-recipes --private

    # once verified, overwrite the real split (all original columns preserved)
    uv run --with datasets python \
        examples/scripts/deployment/augment_source_with_results.py \
        --target-repo LeMaterial/LeMat-Synth-Papers
"""

from __future__ import annotations

import argparse
import json

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
SOURCE_URI = "LeMaterial/LeMat-Synth-Papers"


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="superconductor_keywords_and_LLM",
                    help="config name in BOTH source and results repos")
    ap.add_argument("--split", default="full",
                    help="split name in BOTH source and results repos")
    ap.add_argument("--results-repo",
                    default="sid-betalol/LeMat-Synth-recipes-dummy",
                    help="repo holding the extraction+judge results")
    ap.add_argument("--target-repo", default=None,
                    help="where to PUSH the augmented split. Omit = verify "
                         "only. Use a COPY repo before the real dataset.")
    ap.add_argument("--private", action="store_true",
                    help="push the target repo as private")
    args = ap.parse_args()

    print(f"Loading source: {SOURCE_URI} [{args.config}/{args.split}] ...")
    src = load_dataset(SOURCE_URI, args.config, split=args.split)
    print(f"  source rows: {src.num_rows}, columns: {len(src.column_names)}")

    print(f"Loading results: {args.results_repo} "
          f"[{args.config}/{args.split}] ...")
    res = load_dataset(args.results_repo, args.config, split=args.split)
    # index results by id
    by_id = {}
    for r in res:
        by_id[r["id"]] = r
    print(f"  results rows: {res.num_rows}, indexed ids: {len(by_id)}")

    # Build aligned new-column arrays, joined by id.
    n = src.num_rows
    new_ss, new_eval = [], []
    new_scores = {c: [] for c in SCORE_COLUMNS}
    matched = 0
    for row in src:
        r = by_id.get(row["id"])
        if r is None:
            # no result for this paper: leave recipe/eval null, scores null
            new_ss.append(None)
            new_eval.append(None)
            for c in SCORE_COLUMNS:
                new_scores[c].append(None)
            continue
        matched += 1
        # results already store these as JSON strings; carry through as-is
        ss = r.get("structured_synthesis")
        ev = r.get("evaluations")
        new_ss.append(ss if isinstance(ss, str) else json.dumps(ss))
        new_eval.append(ev if isinstance(ev, str) else json.dumps(ev))
        for c in SCORE_COLUMNS:
            v = r.get(c)
            new_scores[c].append(v if isinstance(v, (int, float)) else None)

    print(f"  matched {matched}/{n} source rows to a result")

    # Replace structured_synthesis (struct -> JSON string) and add new columns.
    out = src.remove_columns(["structured_synthesis"])
    out = out.add_column("structured_synthesis", new_ss)
    out = out.add_column("evaluations", new_eval)
    for c in SCORE_COLUMNS:
        out = out.add_column(c, new_scores[c])

    # --- verification ---
    print("\n=== VERIFICATION ===")
    print(f"rows: {out.num_rows} (source had {n})")
    assert out.num_rows == n, "row count changed!"
    orig_preserved = [
        c for c in src.column_names if c != "structured_synthesis"
    ]
    missing = [c for c in orig_preserved if c not in out.column_names]
    print(f"original columns preserved: {len(orig_preserved)}/"
          f"{len(orig_preserved)} (missing: {missing or 'none'})")
    added = [c for c in out.column_names if c not in src.column_names]
    print(f"new columns added: {added}")
    filled = sum(1 for v in out['structured_synthesis'] if v)
    print(f"structured_synthesis filled: {filled}/{n}")
    scored = sum(1 for v in out['overall_score'] if isinstance(v, (int, float)))
    print(f"overall_score filled: {scored}/{n}")
    print(f"final column count: {len(out.column_names)} "
          f"(source {len(src.column_names)} + {len(added)} new)")

    if not args.target_repo:
        print("\nNo --target-repo: verification only, nothing pushed.")
        return

    print(f"\nPushing to {args.target_repo} "
          f"[{args.config}/{args.split}] private={args.private} ...")
    out.push_to_hub(
        args.target_repo,
        config_name=args.config,
        split=args.split,
        private=args.private,
    )
    print("Pushed parquet. Every original column preserved; "
          "structured_synthesis filled; evaluations + 8 score columns added.")

    # On an existing multi-config dataset, push_to_hub updates the parquet but
    # may leave the dataset card's dataset_info features for this config stale,
    # which breaks load_dataset / the viewer. Patch the card's features block to
    # match the new schema.
    patch_readme_features(
        args.target_repo, args.config, out.features._to_yaml_list()
    )


def _render_features_block(feats):
    """Render a datasets features list as README dataset_info YAML."""
    out = ["  features:"]
    for f in feats:
        out.append(f"  - name: {f['name']}")
        if "dtype" in f:
            out.append(f"    dtype: {f['dtype']}")
        elif isinstance(f.get("list"), str):
            out.append(f"    list: {f['list']}")
        elif isinstance(f.get("list"), list):
            out.append("    list:")
            for sub in f["list"]:
                out.append(f"    - name: {sub['name']}")
                out.append(f"      dtype: {sub['dtype']}")
        else:
            raise ValueError(f"unhandled feature shape: {f}")
    return "\n".join(out)


def patch_readme_features(repo, config, feats):
    """Surgically replace the dataset_info features block for ``config`` in the
    repo README so it matches the pushed parquet schema. No-op if the anchor is
    not found (warns instead of breaking)."""
    from huggingface_hub import HfApi, hf_hub_download

    txt = open(
        hf_hub_download(repo, "README.md", repo_type="dataset")
    ).read()
    anchor = f"- config_name: {config}\n  features:"
    i = txt.find(anchor)
    if i == -1:
        print(f"  [readme] anchor for '{config}' not found; skipping patch. "
              f"If the split fails to load, patch the card manually.")
        return
    feat_start = i + len(f"- config_name: {config}\n")
    rest = txt[feat_start:]
    lines = rest.split("\n")
    pos = len(lines[0]) + 1  # skip the '  features:' line
    end_rel = None
    for ln in lines[1:]:
        # a sibling key (2-space indent, not a features line) ends the block
        if ln.startswith("  ") and not (
            ln.startswith("  -") or ln.startswith("    ")
        ):
            end_rel = pos
            break
        if ln.startswith("- config_name:") or ln.startswith("configs:") or (
            ln and not ln.startswith(" ") and not ln.startswith("-")
        ):
            end_rel = pos
            break
        pos += len(ln) + 1
    if end_rel is None:
        print("  [readme] could not bound features block; skipping patch.")
        return
    new_txt = txt[:feat_start] + _render_features_block(feats) + "\n" \
        + rest[end_rel:]
    if new_txt == txt:
        print("  [readme] features already up to date; no change.")
        return
    HfApi().upload_file(
        path_or_fileobj=new_txt.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"fix: sync dataset_info features for {config}",
    )
    print(f"  [readme] patched dataset_info features for '{config}'.")


if __name__ == "__main__":
    main()
