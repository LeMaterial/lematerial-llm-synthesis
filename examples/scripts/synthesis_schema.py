import pyarrow as pa

schema = pa.schema(
    [
        ("synthesized_material", pa.string()),
        ("material_category", pa.string()),
        ("synthesis_method", pa.string()),
        (
            "images",
            pa.list_(
                pa.struct([("bytes", pa.binary()), ("path", pa.string())])
            ),
        ),
        ("plot_data", pa.null()),
        (
            "structured_synthesis",
            pa.struct(
                [
                    ("target_compound", pa.string()),
                    ("target_compound_type", pa.string()),
                    ("synthesis_method", pa.string()),
                    (
                        "starting_materials",
                        pa.list_(
                            pa.struct(
                                [
                                    ("name", pa.string()),
                                    ("amount", pa.float64()),
                                    ("unit", pa.string()),
                                    ("purity", pa.string()),
                                    ("vendor", pa.string()),
                                ]
                            )
                        ),
                    ),
                    (
                        "steps",
                        pa.list_(
                            pa.struct(
                                [
                                    ("step_number", pa.int64()),
                                    ("action", pa.string()),
                                    ("description", pa.string()),
                                    (
                                        "materials",
                                        pa.list_(
                                            pa.struct(
                                                [
                                                    ("name", pa.string()),
                                                    ("amount", pa.float64()),
                                                    ("unit", pa.string()),
                                                    ("purity", pa.string()),
                                                    ("vendor", pa.string()),
                                                ]
                                            )
                                        ),
                                    ),
                                    (
                                        "equipment",
                                        pa.list_(
                                            pa.struct(
                                                [
                                                    ("name", pa.string()),
                                                    (
                                                        "instrument_vendor",
                                                        pa.string(),
                                                    ),
                                                    ("settings", pa.string()),
                                                ]
                                            )
                                        ),
                                    ),
                                    (
                                        "conditions",
                                        pa.struct(
                                            [
                                                ("temperature", pa.float64()),
                                                ("temp_unit", pa.string()),
                                                ("duration", pa.float64()),
                                                ("time_unit", pa.string()),
                                                ("pressure", pa.float64()),
                                                ("pressure_unit", pa.string()),
                                                ("atmosphere", pa.string()),
                                                ("stirring", pa.bool_()),
                                                (
                                                    "stirring_speed",
                                                    pa.float64(),
                                                ),
                                                ("ph", pa.float64()),
                                            ]
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "equipment",
                        pa.list_(
                            pa.struct(
                                [
                                    ("name", pa.string()),
                                    ("instrument_vendor", pa.string()),
                                    ("settings", pa.string()),
                                ]
                            )
                        ),
                    ),
                    ("notes", pa.string()),
                ]
            ),
        ),
        (
            "evaluation",
            pa.struct(
                [
                    ("reasoning", pa.string()),
                    (
                        "scores",
                        pa.struct(
                            [
                                ("structural_completeness_score", pa.float64()),
                                (
                                    "structural_completeness_reasoning",
                                    pa.string(),
                                ),
                                ("material_extraction_score", pa.float64()),
                                ("material_extraction_reasoning", pa.string()),
                                ("process_steps_score", pa.float64()),
                                ("process_steps_reasoning", pa.string()),
                                ("equipment_extraction_score", pa.float64()),
                                ("equipment_extraction_reasoning", pa.string()),
                                ("conditions_extraction_score", pa.float64()),
                                (
                                    "conditions_extraction_reasoning",
                                    pa.string(),
                                ),
                                ("semantic_accuracy_score", pa.float64()),
                                ("semantic_accuracy_reasoning", pa.string()),
                                ("format_compliance_score", pa.float64()),
                                ("format_compliance_reasoning", pa.string()),
                                ("overall_score", pa.float64()),
                                ("overall_reasoning", pa.string()),
                            ]
                        ),
                    ),
                    ("confidence_level", pa.string()),
                    ("missing_information", pa.list_(pa.string())),
                    ("extraction_errors", pa.list_(pa.string())),
                    ("improvement_suggestions", pa.list_(pa.string())),
                ]
            ),
        ),
        ("synthesis_extraction_performance_llm", pa.int32()),
        ("figure_extraction_performance_llm", pa.int32()),
        ("synthesis_extraction_performance_human", pa.int32()),
        ("figure_extraction_performance_human", pa.int32()),
        ("paper_id", pa.string()),
        ("paper_title", pa.string()),
        ("paper_published_date", pa.string()),
        ("paper_abstract", pa.string()),
        ("paper_doi", pa.string()),
        ("paper_url", pa.string()),
    ]
)
