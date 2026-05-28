"""Validate annotation result_human.json files against the schema.

For every new ``annotations/<id>/result_human.json`` (the ``multi_llm_v1``
files, excluding ``old/``) this checks that each material's ``human_recipe``:

1. passes ``GeneralSynthesisOntology.model_validate`` (types, enum values,
   required fields), and
2. contains no unknown/misnamed field at any nesting level. Pydantic's
   default ``extra="ignore"`` silently drops extras on ``model_validate``,
   so we walk the model tree to catch things like ``durations``,
   ``pressures``, a step-level ``notes``, or ``Instrument_vendor``.

Exits non-zero if any file has issues (usable in CI / pre-commit).

Usage:
    uv run examples/scripts/data_curation/validate_result_human_schema.py
    uv run .../validate_result_human_schema.py --annotations-dir annotations \
        --include-old
"""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path

from pydantic import BaseModel, ValidationError

from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology


def _model_in(annotation) -> type[BaseModel] | None:
    """Return the nested BaseModel referenced by a field annotation, if any.

    Peels ``X | None`` and ``list[X]`` so e.g. ``Conditions | None`` and
    ``list[Material]`` resolve to ``Conditions`` / ``Material``.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = _model_in(arg)
        if found is not None:
            return found
    return None


def find_unknown_fields(
    model_cls: type[BaseModel], data: object, path: str = ""
) -> list[str]:
    """Recursively collect keys in ``data`` not declared on the model."""
    issues: list[str] = []
    if not isinstance(data, dict):
        return issues
    allowed = set(model_cls.model_fields)
    for key in data:
        if key not in allowed:
            issues.append(f"{path or '<root>'}: unknown field '{key}'")
    for name, field in model_cls.model_fields.items():
        if name not in data:
            continue
        nested = _model_in(field.annotation)
        if nested is None:
            continue
        value = data[name]
        child = f"{path}.{name}" if path else name
        if isinstance(value, list):
            for i, item in enumerate(value):
                issues += find_unknown_fields(nested, item, f"{child}[{i}]")
        elif isinstance(value, dict):
            issues += find_unknown_fields(nested, value, child)
    return issues


def iter_recipes(data: object) -> list[tuple[str, object]]:
    """Return (label, recipe) for both multi_llm_v1 and legacy shapes."""
    if isinstance(data, dict):  # multi_llm_v1
        return [
            (m.get("material_name", "<unnamed>"), m.get("human_recipe"))
            for m in data.get("materials", [])
        ]
    if isinstance(data, list):  # legacy flat array
        return [
            (m.get("material", "<unnamed>"), m.get("synthesis")) for m in data
        ]
    return []


def validate_file(path: Path) -> list[str]:
    """Return a list of human-readable issues for one result_human.json."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    recipes = iter_recipes(data)
    if not recipes:
        return ["no materials / human_recipe found"]

    issues: list[str] = []
    for label, recipe in recipes:
        if recipe is None:
            issues.append(f"[{label}] human_recipe missing or null")
            continue
        try:
            GeneralSynthesisOntology.model_validate(recipe)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                issues.append(
                    f"[{label}] {loc}: {err['msg']} (got {err.get('input')!r})"
                )
        for unknown in find_unknown_fields(GeneralSynthesisOntology, recipe):
            issues.append(f"[{label}] {unknown}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate result_human.json files against the schema."
    )
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument(
        "--include-old",
        action="store_true",
        help="Also validate the old/ reference files.",
    )
    args = parser.parse_args()

    root = Path(args.annotations_dir)
    files = sorted(root.glob("*/result_human.json"))
    if args.include_old:
        files += sorted(root.glob("*/old/result_human.json"))

    bad_files = 0
    total_issues = 0
    for path in files:
        issues = validate_file(path)
        if issues:
            bad_files += 1
            total_issues += len(issues)
            print(f"\nFAIL  {path}")
            for issue in issues:
                print(f"        {issue}")

    print("\n" + "=" * 64)
    print(
        f"Checked {len(files)} files: {len(files) - bad_files} clean, "
        f"{bad_files} with issues ({total_issues} issues total)."
    )
    return 1 if bad_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
