"""Compare multi-LLM result.json with result_human.json, LLM-matched variant.

Same as ``compare_multi_llm_results_complete.load_annotations``, except
unmatched human/LLM material names (after the existing threshold-0.7 string
match) get a second pass through ``DspyNameMatcherJudge`` -- the same
LLM-based aligner ``eval_vlm.py`` already uses for the thermocatalysis VLM
digitization eval. Handles cases the string matcher can't, e.g. human
"Bi2-xSbxTe3" vs LLM "Bi2Te3"/"Sb2Te3" (same doped-solid-solution family,
different phrasing), or "HA" vs "Ca10(PO4)6(OH)2" (hydroxyapatite written as
an abbreviation vs. a formula).

Kept as a separate module (not a patch to compare_multi_llm_results_complete.py)
so the original string-only pipeline -- and the CSVs/manuscript numbers it
produced -- stay reproducible as a fallback.

Why this exists / what problem it fixes
----------------------------------------
The string matcher alone (SequenceMatcher + word-Jaccard, threshold 0.7) links
materials in only ~15-17 of the 34 annotated papers per judge -- most of the
rest fail purely on phrasing, not because there's genuinely no correspondence.
This costs real statistical power downstream: e.g. the paper-level bootstrap
CI on judge-agreement metrics (Fig3_Judge_Agreement.ipynb, "Variant 1b"/"1c")
is driven almost entirely by how many independent papers are available, and
15-17 papers gives very wide CIs almost regardless of the true effect size.

Validation (manual review by a domain expert; raw annotations shipped in
``examples/scripts/evaluation/name_matcher_validation/``:
``high_confidence_matches_reviewed.csv``,
``medium_confidence_matches_reviewed.csv``, ``recall_check_reviewed.csv``)
- High-confidence LLM matches: 19/19 manually verified correct (100% precision).
- Medium-confidence LLM matches: 17/19 correct (89% precision). The 2 wrong
  ones are denylisted in ``_REJECTED_MATCHES`` below rather than dropping
  medium confidence wholesale, since 89% precision on ~19 pairs is a
  reasonable bar and the two failures are specific, identifiable errors
  (both dropped a real structural component -- see comment there).
- Low-confidence matches: NOT manually reviewed in aggregate, but the ones
  inspected during development were clearly wrong (e.g. a peptide sequence
  matched to an unrelated molecular formula, or the LLM's own leaked
  reasoning text matched as if it were a material name) -- excluded entirely.
- Recall: checked against a sample of 13 "a match was structurally possible
  but not made" cases (i.e. an LLM-extracted candidate existed for that
  paper but no pair was proposed, or was proposed below the confidence bar).
  4/13 were real misses; 3 of those were medium-confidence matches that got
  filtered out under an earlier (high-only) version of this filter, which is
  why the current filter keeps medium as well as high. Recall was not
  re-measured after that change -- treat the current n as a lower bound on
  how many papers *could* be recovered, not a ceiling.

A caveat surfaced during review that this fix cannot address: some human
ground-truth material names are themselves too vague/inconsistent to link to
any specific LLM extraction even in principle -- e.g. "Pf-AgNPs", "Oxidized
CNS", "Cyanine SMILES" -- because the human annotation guidelines did not
enforce a standardized naming convention (no canonical formula, no required
disambiguation when a paper reports multiple related compounds). No matcher,
string or LLM, can recover a link that was never recorded precisely enough to
resolve. This is a real ceiling on achievable n, not a matching-algorithm bug.

Usage:
    uv run python \\
        examples/scripts/evaluation/compare_multi_llm_results_llm_match.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_multi_llm_results_complete import (  # pylint: disable=wrong-import-position
    SCORE_COLUMNS,
)
from eval_utils import (  # pylint: disable=wrong-import-position
    find_best_matches,
    normalize_material_name,
)

_repo_root_src = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"
)
if _repo_root_src not in sys.path:
    sys.path.insert(0, _repo_root_src)

from llm_synthesis.metrics.judge.name_matcher_judge import (  # noqa: E402
    DspyNameMatcherJudge,
    build_name_match_inputs,
)
from llm_synthesis.utils.llms import LLM_REGISTRY  # noqa: E402

OUTPUT_DIR = "results/agreement_analysis_llm_match"


def build_name_matcher_judge(model_name: str = "claude-sonnet-4.6"):
    """Build a DspyNameMatcherJudge from an LLM_REGISTRY entry (same default
    model eval_vlm.py uses for VLM plot-series/material alignment)."""
    import dspy

    cfg = LLM_REGISTRY.configs[model_name]
    kwargs = dict(cfg.extra_kwargs or {})
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    lm = dspy.LM(cfg.model, temperature=0.1, **kwargs)
    return DspyNameMatcherJudge(lm=lm)


# Manually reviewed and rejected during precision validation (see module
# docstring "Validation" section) -- the LLM judge proposed these at
# confidence=medium, but they drop a real structural component rather than
# just rephrasing the same compound, so they're wrong despite the model's
# confidence. Denylisted by (paper_id, human_name, llm_name) rather than
# lowering the confidence bar further, since medium confidence is otherwise
# ~89% precise (17/19 manually verified correct).
_REJECTED_MATCHES = {
    ("1605.04038", "LaAlO3/SrTiO3", "LaAlO3"),
    (
        "1605.04038",
        "(La0.3Sr0.7)(Al0.65Ta0.35)O3/SrTiO3",
        "(La0.3Sr0.7)(Al0.65Ta0.35)O3",
    ),
}


def _llm_match_unmatched(
    matcher, paper_id, unmatched_human_names, llm_candidates
):
    """LLM-judge fallback: align remaining human names to remaining LLM names.

    Returns dict {human_name: llm_name} for pairs the judge considers a match.
    """
    if not unmatched_human_names or not llm_candidates:
        return {}
    result = matcher.forward(
        build_name_match_inputs(
            unmatched_human_names,
            llm_candidates,
            context=f"paper_id={paper_id}",
        )
    )
    matches = {}
    for pair in result.matches:
        if pair.llm_name is None or pair.llm_name not in llm_candidates:
            continue
        if pair.confidence == "low":
            logging.info(
                "LLM match SKIPPED (confidence=%s) [%s]: %r -> %r",
                pair.confidence,
                paper_id,
                pair.gt_name,
                pair.llm_name,
            )
            continue
        if (paper_id, pair.gt_name, pair.llm_name) in _REJECTED_MATCHES:
            logging.info(
                "LLM match REJECTED (manually verified wrong) [%s]: %r -> %r",
                paper_id,
                pair.gt_name,
                pair.llm_name,
            )
            continue
        matches[pair.gt_name] = pair.llm_name
        logging.info(
            "LLM match [%s]: %r -> %r (confidence=%s)",
            paper_id,
            pair.gt_name,
            pair.llm_name,
            pair.confidence,
        )
    return matches


def load_annotations(annotations_dir, skip_folders=None, matcher=None):
    """Load and align human vs LLM-judge evaluations.

    Identical to compare_multi_llm_results_complete.load_annotations, except
    a second matching pass runs the LLM name matcher (if provided) on
    whatever the string matcher (threshold 0.7) left unmatched.

    Returns:
        (human_df, llm_df) DataFrames with the same columns as the original.
    """
    skip_folders = skip_folders or []
    human_rows, llm_rows = [], []
    processed_papers, skipped_papers, skipped_extractions = [], [], []
    llm_match_count = 0

    for paper_id in sorted(os.listdir(annotations_dir)):
        paper_dir = os.path.join(annotations_dir, paper_id)
        if not os.path.isdir(paper_dir) or paper_id in skip_folders:
            skipped_papers.append(paper_id)
            continue

        llm_path = os.path.join(paper_dir, "result.json")
        human_path = os.path.join(paper_dir, "result_human.json")
        if not (os.path.exists(llm_path) and os.path.exists(human_path)):
            skipped_papers.append(paper_id)
            continue

        try:
            with open(llm_path, encoding="utf-8") as fh:
                llm_data = json.load(fh)
            with open(human_path, encoding="utf-8") as fh:
                human_data = json.load(fh)
        except (json.JSONDecodeError, KeyError) as exc:
            logging.info("Error reading files for %s: %s", paper_id, exc)
            skipped_papers.append(f"{paper_id} (file read error)")
            continue

        processed_papers.append(paper_id)
        n_human_mats = len(human_data.get("materials", []))
        n_synth_llms = len(llm_data)
        logging.info(
            "Processing %s: %d human materials, %d synth LLMs",
            paper_id,
            n_human_mats,
            n_synth_llms,
        )

        extractor_order = human_data.get("extractor_order", [])

        # --- Index LLM materials, judge scores, and raw name mapping ---
        judge_scores_lookup = {}
        raw_name_map = {}
        for entry in llm_data:
            synth_llm = entry.get("synth_llm", "")
            for mat_entry in entry.get("materials", []):
                mat_name = mat_entry.get("material", "")
                synth_info = mat_entry.get("synthesis", {})
                if "Extraction failed:" in str(
                    synth_info.get("notes", "") or ""
                ):
                    skipped_extractions.append(
                        f"{paper_id}/{synth_llm}/{mat_name} (extraction failed)"
                    )
                    continue
                norm_key = (synth_llm, normalize_material_name(mat_name))
                raw_name_map[norm_key] = mat_name
                judge_scores_lookup[norm_key] = {
                    evaluation.get("judge_llm", ""): evaluation.get(
                        "evaluation", {}
                    ).get("scores", {})
                    for evaluation in mat_entry.get("evaluations", [])
                }

        # --- Collect human scores per synth LLM (first pass) ---
        human_scores_by_synth = {}
        human_category_by_mat = {}
        for human_mat in human_data.get("materials", []):
            mat_name = human_mat.get("material_name", "")
            recipe = human_mat.get("human_recipe") or {}
            human_category_by_mat[mat_name] = {
                "target_compound_type": recipe.get("target_compound_type"),
                "synthesis_method": recipe.get("synthesis_method"),
            }
            evals = human_mat.get("evaluations", [])
            for idx, synth_llm in enumerate(extractor_order):
                if idx >= len(evals):
                    continue
                scores = evals[idx].get("evaluation", {}).get("scores", {})
                if not any(scores.get(c) is not None for c in SCORE_COLUMNS):
                    continue
                human_scores_by_synth.setdefault(synth_llm, {})[mat_name] = (
                    scores
                )

        # --- Greedy string best-match per synth LLM (threshold 0.7) ---
        match_map = {}  # synth_llm -> {human_name: (synth_llm, norm_llm_name)}
        for synth_llm, h_scores in human_scores_by_synth.items():
            h_names = list(h_scores.keys())
            norm_h = [normalize_material_name(n) for n in h_names]
            norm_l = [nk for (sl, nk) in judge_scores_lookup if sl == synth_llm]
            matches = find_best_matches(
                norm_h, norm_l, similarity_threshold=0.7
            )
            norm_to_orig = {
                normalize_material_name(h_name): h_name for h_name in h_names
            }
            synth_match_map = {
                norm_to_orig[norm_human_name]: (synth_llm, norm_llm_name)
                for norm_human_name, norm_llm_name in matches.items()
                if norm_human_name in norm_to_orig
            }

            # --- LLM-judge fallback for names the string matcher missed ---
            if matcher is not None:
                unmatched_human = [
                    h for h in h_names if h not in synth_match_map
                ]
                used_norm_llm = set(
                    synth_match_map[h][1] for h in synth_match_map
                )
                remaining_llm_norm = [
                    nk for nk in norm_l if nk not in used_norm_llm
                ]
                # present original (non-normalized) names to the LLM judge --
                # normalized strings ("bi2-xsbxte3") lose chemistry cues
                # it needs
                llm_orig_candidates = {
                    raw_name_map[(synth_llm, nk)]: nk
                    for nk in remaining_llm_norm
                }
                llm_matches = _llm_match_unmatched(
                    matcher,
                    paper_id,
                    unmatched_human,
                    list(llm_orig_candidates),
                )
                for h_name, llm_orig_name in llm_matches.items():
                    synth_match_map[h_name] = (
                        synth_llm,
                        llm_orig_candidates[llm_orig_name],
                    )
                llm_match_count += len(llm_matches)

            match_map[synth_llm] = synth_match_map

        # --- Build DataFrame rows (second pass) ---
        matched_by_synth = {}
        human_only_by_synth = {}
        matched_llm_keys = set()

        for synth_llm, h_scores in human_scores_by_synth.items():
            matched_by_synth[synth_llm] = []
            human_only_by_synth[synth_llm] = []

            for mat_name, scores in h_scores.items():
                material_id = f"{paper_id}__{synth_llm}__{mat_name}"
                category = human_category_by_mat.get(mat_name, {})
                base = {
                    "paper_id": paper_id,
                    "material_id": material_id,
                    "material": mat_name,
                    "synth_llm": synth_llm,
                    "target_compound_type": category.get(
                        "target_compound_type"
                    ),
                    "synthesis_method": category.get("synthesis_method"),
                }
                lookup_key = match_map.get(synth_llm, {}).get(mat_name)

                human_rows.append(
                    {
                        **base,
                        "judge_id": "human",
                        **{c: scores.get(c) for c in SCORE_COLUMNS},
                    }
                )

                if not lookup_key:
                    human_only_by_synth[synth_llm].append(mat_name)
                    continue

                matched_norm = lookup_key[1]
                orig_llm_name = raw_name_map.get(lookup_key, matched_norm)
                display = (
                    mat_name
                    if normalize_material_name(mat_name) == matched_norm
                    else f"{mat_name} -> {orig_llm_name}"
                )
                matched_by_synth[synth_llm].append(display)
                matched_llm_keys.add(lookup_key)

                for judge_llm, j_scores in judge_scores_lookup[
                    lookup_key
                ].items():
                    llm_rows.append(
                        {
                            **base,
                            "judge_id": judge_llm,
                            **{c: j_scores.get(c) for c in SCORE_COLUMNS},
                        }
                    )

        llm_only_by_synth = {}
        for (sl, nk), raw in raw_name_map.items():
            if (sl, nk) not in matched_llm_keys:
                llm_only_by_synth.setdefault(sl, []).append(raw)

        all_synths = (
            set(matched_by_synth)
            | set(human_only_by_synth)
            | set(llm_only_by_synth)
        )
        for synth_llm in sorted(all_synths):
            matched = matched_by_synth.get(synth_llm, [])
            human_only = human_only_by_synth.get(synth_llm, [])
            llm_only = llm_only_by_synth.get(synth_llm, [])
            total_human = len(matched) + len(human_only)
            if not (matched or human_only or llm_only):
                continue
            if total_human == 0:
                logging.info("  [%s] No human evaluations", synth_llm)
            else:
                logging.info(
                    "  [%s] %d/%d human materials matched",
                    synth_llm,
                    len(matched),
                    total_human,
                )
            if matched:
                logging.info("    Matched: %s", matched)
            if human_only:
                logging.info("    Unmatched (human-only): %s", human_only)
            if llm_only:
                logging.info("    Unmatched (llm-only): %s", llm_only)

    logging.info(
        "\nProcessed %d papers with both human and LLM evaluations:",
        len(processed_papers),
    )
    for paper in processed_papers:
        logging.info("  - %s", paper)
    if skipped_papers:
        logging.info("\nSkipped %d papers:", len(skipped_papers))
        for paper in skipped_papers:
            logging.info("  - %s", paper)
    if skipped_extractions:
        logging.info(
            "\nSkipped %d materials (extraction failures):",
            len(skipped_extractions),
        )
        for item in skipped_extractions:
            logging.info("  - %s", item)
    logging.info(
        "\nTotal human rows: %d | Total LLM rows: %d | LLM-judge matches: %d",
        len(human_rows),
        len(llm_rows),
        llm_match_count,
    )

    return pd.DataFrame(human_rows), pd.DataFrame(llm_rows)
