"""
LeMat-Synth Human Annotation App

A Streamlit UI for the 5-step annotation workflow:
  1. Pick a paper
  2. Read the PDF
  3. Fill in the human recipe
  4. Score each LLM extraction (blind)
  5. Save result_human.json

Run with:  streamlit run examples/scripts/data_curation/annotator_app.py
"""

import copy
import json
from pathlib import Path

import streamlit as st

ANNOTATIONS_DIR = Path("annotations")
SKIP_FOLDERS = {"annotation_guide_catalysis"}

COMPOUND_TYPES = [
    "",
    "ceramics & glasses",
    "metals & alloys",
    "semiconductors & electronic",
    "functional materials & catalysts",
    "polymers & composites",
    "nanomaterials",
    "thin films & coatings",
    "other",
]

SCORE_DIMS = [
    ("structural_completeness", "Structural Completeness", "Are all fields populated that should be?"),
    ("material_extraction", "Material Extraction", "Correct compounds, amounts, purities?"),
    ("process_steps", "Process Steps", "Steps complete, in right order?"),
    ("equipment_extraction", "Equipment Extraction", "Equipment correctly identified?"),
    ("conditions_extraction", "Conditions Extraction", "Temps, durations, atmosphere captured?"),
    ("semantic_accuracy", "Semantic Accuracy", "Right action verbs and terminology?"),
    ("format_compliance", "Format Compliance", "Follows the schema correctly?"),
    ("overall", "Overall", "Overall quality of the extraction"),
]

SCORE_OPTIONS = [None, 0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

SM_FIELDS = ["name", "vendor", "amount", "unit", "purity"]
EQ_FIELDS = ["name", "instrument_vendor", "settings"]
COND_NUM_FIELDS = ["temperature", "duration", "pressure", "ph"]
COND_STR_FIELDS = ["temp_unit", "time_unit", "pressure_unit", "atmosphere"]

def get_paper_ids():
    return sorted(
        d.name
        for d in ANNOTATIONS_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_FOLDERS and not d.name.startswith(".")
    )

def get_status(paper_id):
    rh = ANNOTATIONS_DIR / paper_id / "result_human.json"
    if not rh.exists():
        return "no_file"
    data = json.loads(rh.read_text(encoding="utf-8"))
    mats = data.get("materials", [])
    has_recipe = any(m.get("human_recipe", {}).get("target_compound") for m in mats)
    has_scores = any(
        ev.get("evaluation", {}).get("scores", {}).get("structural_completeness_score") is not None
        for m in mats
        for ev in m.get("evaluations", [])
    )
    if has_recipe and has_scores:
        return "complete"
    if has_recipe:
        return "recipe_only"
    return "empty"

def load_data(paper_id):
    rh = ANNOTATIONS_DIR / paper_id / "result_human.json"
    r = ANNOTATIONS_DIR / paper_id / "result.json"
    human = json.loads(rh.read_text(encoding="utf-8"))
    llm = json.loads(r.read_text(encoding="utf-8"))
    return human, llm

def load_old_annotation(paper_id):
    old = ANNOTATIONS_DIR / paper_id / "old" / "result_human.json"
    if old.exists():
        return json.loads(old.read_text(encoding="utf-8"))
    return None

def _noe(val):
    if val is None:
        return None
    if isinstance(val, str) and val.strip() == "":
        return None
    return val

def _parse_float(val):
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return None if f == 0.0 else f
    except (ValueError, TypeError):
        return None

def _normalize_score(val):
    if val is None:
        return None
    try:
        f = float(val)
        if f in SCORE_OPTIONS:
            return f
        return round(f * 2) / 2
    except (ValueError, TypeError):
        return None

def materials_to_text(materials):
    lines = []
    for m in materials or []:
        parts = []
        name = m.get("name") or ""
        if not name:
            continue
        parts.append(name)
        if m.get("amount") is not None:
            parts.append(f"({m['amount']} {m.get('unit') or ''})")
        if m.get("vendor"):
            parts.append(f"[vendor: {m['vendor']}]")
        if m.get("purity"):
            parts.append(f"[purity: {m['purity']}]")
        lines.append(" ".join(parts))
    return "\n".join(lines)

def text_to_materials(text):
    result = []
    for line in (text or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        result.append({"name": line, "vendor": None, "amount": None, "unit": None, "purity": None})
    return result

def equipment_to_text(equipment):
    lines = []
    for e in equipment or []:
        name = e.get("name") or ""
        if not name:
            continue
        parts = [name]
        vendor = e.get("instrument_vendor") or ""
        settings = e.get("settings") or ""
        if vendor or settings:
            parts.append(vendor)
        if settings:
            parts.append(settings)
        lines.append(" | ".join(parts))
    return "\n".join(lines)

def text_to_equipment(text):
    result = []
    for line in (text or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        result.append({
            "name": parts[0] if parts else "",
            "instrument_vendor": _noe(parts[1]) if len(parts) > 1 else None,
            "settings": _noe(parts[2]) if len(parts) > 2 else None,
        })
    return result

def init_state(paper_id, human_data, mat_idx=0):
    st.session_state._paper = paper_id
    st.session_state._mat_idx = mat_idx
    st.session_state._human_base = copy.deepcopy(human_data)

    mat = human_data["materials"][mat_idx]
    recipe = mat.get("human_recipe", {})

    st.session_state.w_target = recipe.get("target_compound") or ""
    st.session_state.w_type = recipe.get("target_compound_type") or ""
    st.session_state.w_method = recipe.get("synthesis_method") or ""
    st.session_state.w_notes = recipe.get("notes") or ""

    sms = recipe.get("starting_materials") or []
    st.session_state.n_sm = len(sms)
    for i, m in enumerate(sms):
        st.session_state[f"sm_{i}_name"] = m.get("name") or ""
        st.session_state[f"sm_{i}_vendor"] = m.get("vendor") or ""
        st.session_state[f"sm_{i}_amount"] = str(m["amount"]) if m.get("amount") is not None else ""
        st.session_state[f"sm_{i}_unit"] = m.get("unit") or ""
        st.session_state[f"sm_{i}_purity"] = m.get("purity") or ""

    steps = recipe.get("steps") or []
    st.session_state.n_steps = len(steps)
    for i, s in enumerate(steps):
        st.session_state[f"step_{i}_action"] = s.get("action") or ""
        st.session_state[f"step_{i}_desc"] = s.get("description") or ""

        cond = s.get("conditions") or {}
        for f in COND_NUM_FIELDS:
            val = cond.get(f)
            st.session_state[f"step_{i}_c_{f}"] = str(val) if val is not None else ""
        for f in COND_STR_FIELDS:
            st.session_state[f"step_{i}_c_{f}"] = cond.get(f) or ""

        st.session_state[f"step_{i}_mats_text"] = materials_to_text(s.get("materials"))
        st.session_state[f"step_{i}_eq_text"] = equipment_to_text(s.get("equipment"))

    equips = recipe.get("equipment") or []
    st.session_state.n_eq = len(equips)
    for i, e in enumerate(equips):
        st.session_state[f"eq_{i}_name"] = e.get("name") or ""
        st.session_state[f"eq_{i}_vendor"] = e.get("instrument_vendor") or ""
        st.session_state[f"eq_{i}_settings"] = e.get("settings") or ""

    evals = mat.get("evaluations") or []
    for i in range(4):
        ev = evals[i].get("evaluation", {}) if i < len(evals) else {}
        scores = ev.get("scores", {})

        for dim_key, _, _ in SCORE_DIMS:
            raw = scores.get(f"{dim_key}_score")
            st.session_state[f"ev_{i}_{dim_key}_s"] = _normalize_score(raw)
            st.session_state[f"ev_{i}_{dim_key}_r"] = scores.get(f"{dim_key}_reasoning") or ""

        st.session_state[f"ev_{i}_conf"] = ev.get("confidence_level")
        st.session_state[f"ev_{i}_reason"] = ev.get("reasoning") or ""
        st.session_state[f"ev_{i}_missing"] = "\n".join(ev.get("missing_information") or [])
        st.session_state[f"ev_{i}_errors"] = "\n".join(ev.get("extraction_errors") or [])
        st.session_state[f"ev_{i}_suggestions"] = "\n".join(ev.get("improvement_suggestions") or [])

def collect_output():
    output = copy.deepcopy(st.session_state._human_base)
    mat_idx = st.session_state._mat_idx
    mat = output["materials"][mat_idx]
    recipe = mat["human_recipe"]

    recipe["target_compound"] = _noe(st.session_state.get("w_target"))
    recipe["target_compound_type"] = _noe(st.session_state.get("w_type"))
    recipe["synthesis_method"] = _noe(st.session_state.get("w_method"))
    recipe["notes"] = _noe(st.session_state.get("w_notes"))

    recipe["starting_materials"] = []
    for i in range(st.session_state.get("n_sm", 0)):
        name = st.session_state.get(f"sm_{i}_name", "")
        if not name.strip():
            continue
        recipe["starting_materials"].append({
            "name": _noe(name),
            "vendor": _noe(st.session_state.get(f"sm_{i}_vendor")),
            "amount": _parse_float(st.session_state.get(f"sm_{i}_amount")),
            "unit": _noe(st.session_state.get(f"sm_{i}_unit")),
            "purity": _noe(st.session_state.get(f"sm_{i}_purity")),
        })

    recipe["steps"] = []
    for i in range(st.session_state.get("n_steps", 0)):
        cond = {}
        for f in COND_NUM_FIELDS:
            cond[f] = _parse_float(st.session_state.get(f"step_{i}_c_{f}"))
        for f in COND_STR_FIELDS:
            cond[f] = _noe(st.session_state.get(f"step_{i}_c_{f}"))
        cond["stirring"] = None
        cond["stirring_speed"] = None

        step_mats = text_to_materials(st.session_state.get(f"step_{i}_mats_text", ""))
        step_eq = text_to_equipment(st.session_state.get(f"step_{i}_eq_text", ""))

        recipe["steps"].append({
            "step_number": i + 1,
            "action": _noe(st.session_state.get(f"step_{i}_action")),
            "description": _noe(st.session_state.get(f"step_{i}_desc")),
            "materials": step_mats,
            "equipment": step_eq,
            "conditions": cond,
        })

    recipe["equipment"] = []
    for i in range(st.session_state.get("n_eq", 0)):
        name = st.session_state.get(f"eq_{i}_name", "")
        if not name.strip():
            continue
        recipe["equipment"].append({
            "name": _noe(name),
            "instrument_vendor": _noe(st.session_state.get(f"eq_{i}_vendor")),
            "settings": _noe(st.session_state.get(f"eq_{i}_settings")),
        })

    for i in range(4):
        if i >= len(mat["evaluations"]):
            break
        scores = {}
        for dim_key, _, _ in SCORE_DIMS:
            raw = st.session_state.get(f"ev_{i}_{dim_key}_s")
            scores[f"{dim_key}_score"] = _normalize_score(raw)
            scores[f"{dim_key}_reasoning"] = st.session_state.get(f"ev_{i}_{dim_key}_r", "")

        def _lines(key):
            return [x.strip() for x in st.session_state.get(key, "").split("\n") if x.strip()]

        mat["evaluations"][i]["evaluation"] = {
            "reasoning": st.session_state.get(f"ev_{i}_reason", ""),
            "scores": scores,
            "confidence_level": st.session_state.get(f"ev_{i}_conf"),
            "missing_information": _lines(f"ev_{i}_missing"),
            "extraction_errors": _lines(f"ev_{i}_errors"),
            "improvement_suggestions": _lines(f"ev_{i}_suggestions"),
        }

    return output

def format_extraction_md(synthesis):
    if not synthesis:
        return "*No extraction data*"

    lines = []
    lines.append(f"**Target:** {synthesis.get('target_compound', 'N/A')}")
    lines.append(f"**Type:** {synthesis.get('target_compound_type', 'N/A')}")
    lines.append(f"**Method:** {synthesis.get('synthesis_method', 'N/A')}")

    sms = synthesis.get("starting_materials") or []
    if sms:
        lines.append("\n**Starting Materials:**")
        for m in sms:
            parts = [m.get("name") or "?"]
            if m.get("amount") is not None:
                parts.append(f"({m['amount']} {m.get('unit') or ''})")
            if m.get("vendor"):
                parts.append(f"[vendor: {m['vendor']}]")
            if m.get("purity"):
                parts.append(f"[purity: {m['purity']}]")
            lines.append(f"- {' '.join(parts)}")

    steps = synthesis.get("steps") or []
    if steps:
        lines.append("\n**Steps:**")
        for s in steps:
            num = s.get("step_number", "?")
            action = s.get("action", "?")
            lines.append(f"\n**Step {num} - {action}**")
            desc = s.get("description", "")
            if desc:
                lines.append(f"> {desc}")

            step_mats = s.get("materials") or []
            if step_mats:
                names = ", ".join(m.get("name", "?") for m in step_mats)
                lines.append(f"- *Materials:* {names}")

            step_eq = s.get("equipment") or []
            for e in step_eq:
                eq_str = e.get("name", "?")
                if e.get("settings"):
                    eq_str += f" ({e['settings']})"
                lines.append(f"- *Equipment:* {eq_str}")

            cond = s.get("conditions")
            if cond and isinstance(cond, dict):
                cond_parts = []
                if cond.get("temperature") is not None:
                    cond_parts.append(f"{cond['temperature']} {cond.get('temp_unit') or ''}")
                if cond.get("duration") is not None:
                    cond_parts.append(f"{cond['duration']} {cond.get('time_unit') or ''}")
                if cond.get("pressure") is not None:
                    cond_parts.append(f"P={cond['pressure']} {cond.get('pressure_unit') or ''}")
                if cond.get("atmosphere"):
                    cond_parts.append(f"atm: {cond['atmosphere']}")
                if cond_parts:
                    lines.append(f"- *Conditions:* {', '.join(cond_parts)}")

    equip = synthesis.get("equipment") or []
    if equip:
        lines.append("\n**Equipment:**")
        for e in equip:
            eq_str = e.get("name", "?")
            if e.get("instrument_vendor"):
                eq_str += f" (vendor: {e['instrument_vendor']})"
            if e.get("settings"):
                eq_str += f" [{e['settings']}]"
            lines.append(f"- {eq_str}")

    if synthesis.get("notes"):
        lines.append(f"\n**Notes:** {synthesis['notes']}")

    return "\n".join(lines)

def _remove_sm(idx):
    n = st.session_state.n_sm
    for j in range(idx, n - 1):
        for f in SM_FIELDS:
            st.session_state[f"sm_{j}_{f}"] = st.session_state.get(f"sm_{j + 1}_{f}", "")
    for f in SM_FIELDS:
        st.session_state.pop(f"sm_{n - 1}_{f}", None)
    st.session_state.n_sm -= 1

def _add_sm():
    i = st.session_state.n_sm
    for f in SM_FIELDS:
        st.session_state[f"sm_{i}_{f}"] = ""
    st.session_state.n_sm += 1

def _remove_eq(idx):
    n = st.session_state.n_eq
    for j in range(idx, n - 1):
        for f in EQ_FIELDS:
            st.session_state[f"eq_{j}_{f}"] = st.session_state.get(f"eq_{j + 1}_{f}", "")
    for f in EQ_FIELDS:
        st.session_state.pop(f"eq_{n - 1}_{f}", None)
    st.session_state.n_eq -= 1

def _add_eq():
    i = st.session_state.n_eq
    for f in EQ_FIELDS:
        st.session_state[f"eq_{i}_{f}"] = ""
    st.session_state.n_eq += 1

def _add_step():
    i = st.session_state.n_steps
    st.session_state[f"step_{i}_action"] = ""
    st.session_state[f"step_{i}_desc"] = ""
    for f in COND_NUM_FIELDS + COND_STR_FIELDS:
        st.session_state[f"step_{i}_c_{f}"] = ""
    st.session_state[f"step_{i}_mats_text"] = ""
    st.session_state[f"step_{i}_eq_text"] = ""
    st.session_state.n_steps += 1

def _remove_step(idx):
    n = st.session_state.n_steps
    step_keys = (
        ["action", "desc", "mats_text", "eq_text"]
        + [f"c_{f}" for f in COND_NUM_FIELDS]
        + [f"c_{f}" for f in COND_STR_FIELDS]
    )
    for j in range(idx, n - 1):
        for k in step_keys:
            st.session_state[f"step_{j}_{k}"] = st.session_state.get(f"step_{j + 1}_{k}", "")
    for k in step_keys:
        st.session_state.pop(f"step_{n - 1}_{k}", None)
    st.session_state.n_steps -= 1

def render_recipe_form():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Target Compound", key="w_target")
    with col2:
        types = COMPOUND_TYPES
        current = st.session_state.get("w_type", "")
        idx = types.index(current) if current in types else 0
        st.selectbox("Compound Type", types, index=idx, key="w_type")
    with col3:
        st.text_input("Synthesis Method", key="w_method")

    st.subheader("Starting Materials")
    n_sm = st.session_state.get("n_sm", 0)

    if n_sm > 0:
        header = st.columns([3, 2, 1.5, 1.5, 1.5, 0.5])
        labels = ["Name", "Vendor", "Amount", "Unit", "Purity", ""]
        for col, label in zip(header, labels):
            col.caption(label)

    for i in range(n_sm):
        cols = st.columns([3, 2, 1.5, 1.5, 1.5, 0.5])
        with cols[0]:
            st.text_input("name", key=f"sm_{i}_name", label_visibility="collapsed")
        with cols[1]:
            st.text_input("vendor", key=f"sm_{i}_vendor", label_visibility="collapsed")
        with cols[2]:
            st.text_input("amount", key=f"sm_{i}_amount", label_visibility="collapsed")
        with cols[3]:
            st.text_input("unit", key=f"sm_{i}_unit", label_visibility="collapsed")
        with cols[4]:
            st.text_input("purity", key=f"sm_{i}_purity", label_visibility="collapsed")
        with cols[5]:
            st.button("X", key=f"sm_del_{i}", on_click=_remove_sm, args=(i,))

    st.button("+ Add Starting Material", key="sm_add_btn", on_click=_add_sm)

    st.subheader("Synthesis Steps")
    n_steps = st.session_state.get("n_steps", 0)

    for i in range(n_steps):
        action_val = st.session_state.get(f"step_{i}_action", "")
        desc_val = st.session_state.get(f"step_{i}_desc", "")
        label = f"Step {i + 1}: {action_val or '(no action)'}"
        if desc_val:
            label += f" -- {desc_val[:50]}..."

        with st.expander(label, expanded=(n_steps <= 3)):
            c1, c2 = st.columns([2, 6])
            with c1:
                st.text_input("Action verb", key=f"step_{i}_action")
            with c2:
                st.text_area("Description", key=f"step_{i}_desc", height=80)

            st.caption("Conditions")
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.text_input("Temperature", key=f"step_{i}_c_temperature", placeholder="e.g. 800")
                st.text_input("Temp unit", key=f"step_{i}_c_temp_unit", placeholder="C")
            with cc2:
                st.text_input("Duration", key=f"step_{i}_c_duration", placeholder="e.g. 36")
                st.text_input("Time unit", key=f"step_{i}_c_time_unit", placeholder="hour")
            with cc3:
                st.text_input("Pressure", key=f"step_{i}_c_pressure", placeholder="e.g. 1e-6")
                st.text_input("Pressure unit", key=f"step_{i}_c_pressure_unit", placeholder="Torr")
            with cc4:
                st.text_input("Atmosphere", key=f"step_{i}_c_atmosphere", placeholder="e.g. argon")
                st.text_input("pH", key=f"step_{i}_c_ph", placeholder="")

            st.caption("Materials used in this step (one per line)")
            st.text_area("Step materials", key=f"step_{i}_mats_text", height=60, label_visibility="collapsed")
            st.caption("Equipment for this step (format: name | vendor | settings, one per line)")
            st.text_area("Step equipment", key=f"step_{i}_eq_text", height=60, label_visibility="collapsed")

            st.button(f"Remove step {i + 1}", key=f"step_del_{i}", on_click=_remove_step, args=(i,))

    st.button("+ Add Step", key="step_add_btn", on_click=_add_step)

    st.subheader("Equipment (top-level)")
    n_eq = st.session_state.get("n_eq", 0)

    if n_eq > 0:
        header = st.columns([3, 2, 3, 0.5])
        for col, label in zip(header, ["Name", "Vendor", "Settings", ""]):
            col.caption(label)

    for i in range(n_eq):
        cols = st.columns([3, 2, 3, 0.5])
        with cols[0]:
            st.text_input("name", key=f"eq_{i}_name", label_visibility="collapsed")
        with cols[1]:
            st.text_input("vendor", key=f"eq_{i}_vendor", label_visibility="collapsed")
        with cols[2]:
            st.text_input("settings", key=f"eq_{i}_settings", label_visibility="collapsed")
        with cols[3]:
            st.button("X", key=f"eq_del_{i}", on_click=_remove_eq, args=(i,))

    st.button("+ Add Equipment", key="eq_add_btn", on_click=_add_eq)

    st.text_area("Notes", key="w_notes", height=100)

def render_scoring_tabs(llm_data, human_data, mat_idx):
    extractor_order = human_data.get("extractor_order", [])
    material_name = human_data["materials"][mat_idx].get("material_name", "")

    llm_by_name = {entry["synth_llm"]: entry for entry in llm_data}

    tabs = st.tabs([f"Extractor {i + 1}" for i in range(4)])

    for i, tab in enumerate(tabs):
        with tab:
            if i < len(extractor_order):
                llm_name = extractor_order[i]
                llm_entry = llm_by_name.get(llm_name, {})
                llm_materials = llm_entry.get("materials", [])
                normalized_target = material_name.lower().strip()

                best_match = None
                for lm in llm_materials:
                    candidate_normalized = (lm.get("material") or "").lower().strip()
                    if candidate_normalized == normalized_target:
                        best_match = lm
                        break

                if best_match:
                    with st.container(border=True):
                        st.markdown("#### LLM Extraction")
                        st.caption(f"Matched material: {best_match.get('material') or '(unnamed)'}")
                        st.markdown(format_extraction_md(best_match.get("synthesis", {})))
                    with st.expander("Raw JSON", expanded=False):
                        st.json(best_match)
                elif len(llm_materials) > 1:
                    st.warning(
                        f"No exact match for '{material_name}'. This extractor returned {len(llm_materials)} materials."
                    )
                    st.caption("Review each extracted material below and score based on your judgment.")
                    for j, lm in enumerate(llm_materials, start=1):
                        extracted_name = lm.get("material") or "(unnamed)"
                        with st.expander(f"Extracted material {j}: {extracted_name}", expanded=(j == 1)):
                            st.markdown(format_extraction_md(lm.get("synthesis", {})))
                    with st.expander("Raw JSON (all materials)", expanded=False):
                        st.json(llm_materials)
                elif len(llm_materials) == 1:
                    only_material = llm_materials[0]
                    st.warning(f"No exact match for '{material_name}'. Showing the only extracted material.")
                    with st.container(border=True):
                        st.markdown("#### LLM Extraction")
                        st.caption(f"Extracted material: {only_material.get('material') or '(unnamed)'}")
                        st.markdown(format_extraction_md(only_material.get("synthesis", {})))
                    with st.expander("Raw JSON", expanded=False):
                        st.json(only_material)
                else:
                    st.warning("No extraction found for this material from this extractor.")
            else:
                st.warning("Extractor order not defined for this index.")

            st.divider()
            st.markdown("#### Your Scores (1-5 scale)")

            for dim_key, dim_label, dim_hint in SCORE_DIMS:
                cols = st.columns([3, 5])
                with cols[0]:
                    st.select_slider(
                        dim_label,
                        options=SCORE_OPTIONS,
                        key=f"ev_{i}_{dim_key}_s",
                        format_func=lambda x: "N/A" if x is None else f"{x:.1f}",
                        help=dim_hint,
                    )
                with cols[1]:
                    st.text_input(
                        f"{dim_label} reasoning",
                        key=f"ev_{i}_{dim_key}_r",
                        placeholder=dim_hint,
                        label_visibility="collapsed",
                    )

            st.divider()

            st.select_slider(
                "Confidence Level",
                options=[None, 1, 2, 3, 4, 5],
                key=f"ev_{i}_conf",
                format_func=lambda x: "N/A" if x is None else str(x),
            )

            st.text_area("Overall Reasoning", key=f"ev_{i}_reason", height=80)
            st.text_area("Missing Information (one per line)", key=f"ev_{i}_missing", height=80)
            st.text_area("Extraction Errors (one per line)", key=f"ev_{i}_errors", height=80)
            st.text_area("Improvement Suggestions (one per line)", key=f"ev_{i}_suggestions", height=80)

def render_save(paper_id):
    output = collect_output()
    mat_idx = st.session_state._mat_idx
    mat = output["materials"][mat_idx]

    recipe = mat["human_recipe"]
    warnings = []
    if not recipe.get("target_compound"):
        warnings.append("Target compound is empty")
    if not recipe.get("synthesis_method"):
        warnings.append("Synthesis method is empty")
    if not recipe.get("steps"):
        warnings.append("No synthesis steps defined")

    scored_extractors = 0
    for i, ev in enumerate(mat.get("evaluations", [])):
        scores = ev.get("evaluation", {}).get("scores", {})
        has_any = any(scores.get(f"{d[0]}_score") is not None for d in SCORE_DIMS)
        if has_any:
            scored_extractors += 1
        else:
            warnings.append(f"Extractor {i + 1} has no scores")

    col1, col2 = st.columns([1, 1])
    with col1:
        if warnings:
            for w in warnings:
                st.warning(w, icon="\u26a0\ufe0f")
        else:
            st.success("All sections filled!")

        st.metric("Extractors scored", f"{scored_extractors}/4")

    with col2:
        with st.expander("Preview JSON", expanded=False):
            st.json(output)

    if st.button("Save to File", type="primary", use_container_width=True):
        path = ANNOTATIONS_DIR / paper_id / "result_human.json"
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        st.session_state._human_base = copy.deepcopy(output)
        st.success(f"Saved to {path}")

def main():
    st.set_page_config(page_title="LeMat-Synth Annotator", layout="wide", page_icon="\U0001f52c")

    papers = get_paper_ids()
    statuses = {p: get_status(p) for p in papers}

    st.sidebar.title("LeMat-Synth Annotator")

    complete = sum(1 for s in statuses.values() if s == "complete")
    recipe_only = sum(1 for s in statuses.values() if s == "recipe_only")
    empty = len(papers) - complete - recipe_only

    st.sidebar.progress(
        complete / max(len(papers), 1),
        text=f"Complete: {complete}/{len(papers)}",
    )
    st.sidebar.caption(f"Recipe only: {recipe_only} | Empty: {empty}")

    status_icons = {"complete": "\u2705", "recipe_only": "\U0001f7e1", "empty": "\U0001f534", "no_file": "\u26aa"}

    selected = st.sidebar.selectbox(
        "Select Paper",
        papers,
        format_func=lambda p: f"{status_icons.get(statuses.get(p, ''), '?')} {p}",
        key="paper_sel",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Legend**")
    st.sidebar.markdown("\u2705 Complete &nbsp; \U0001f7e1 Recipe only &nbsp; \U0001f534 Empty")

    human_data, llm_data = load_data(selected)

    materials = human_data.get("materials", [])
    mat_idx = 0
    if len(materials) > 1:
        mat_names = [m.get("material_name", f"Material {i + 1}") for i, m in enumerate(materials)]
        mat_idx = st.sidebar.selectbox(
            "Material",
            range(len(mat_names)),
            format_func=lambda i: mat_names[i],
            key="mat_sel",
        )

    if st.session_state.get("_paper") != selected or st.session_state.get("_mat_idx") != mat_idx:
        init_state(selected, human_data, mat_idx)
        st.rerun()

    material_name = materials[mat_idx].get("material_name", "Unknown") if materials else "Unknown"
    st.title(f"{selected}")
    st.caption(f"Material: **{material_name}** | Status: {status_icons.get(statuses.get(selected, ''), '?')} {statuses.get(selected, '?')}")

    paper_url = human_data.get("paper_url", "")
    with st.expander("Paper PDF", expanded=False):
        if paper_url:
            st.markdown(f"[Open PDF in new tab]({paper_url})")
            st.components.v1.iframe(paper_url, height=800, scrolling=True)
        else:
            st.warning("No PDF URL available for this paper.")

    old = load_old_annotation(selected)
    if old:
        with st.expander("Reference: Old annotation (previous schema)", expanded=False):
            st.json(old)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.header("Human Recipe")
        render_recipe_form()

    with right:
        st.header("Score LLM Extractions (blind)")
        render_scoring_tabs(llm_data, human_data, mat_idx)

    st.divider()

    st.header("Save")
    render_save(selected)

if __name__ == "__main__":
    main()
