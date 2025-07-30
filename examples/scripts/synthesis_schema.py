import pyarrow as pa

schema = pa.schema([
    ('synthesized_material', pa.string()),
    ('material_category', pa.string()),
    ('synthesis_method', pa.string()),    
    ('images', pa.list_(pa.struct([('bytes', pa.binary()), ('path', pa.string())]))),
    ('structured_synthesis', pa.struct([
        ('target_compound', pa.string()),
        ('synthesis_method', pa.string()), 
        ('starting_materials', pa.list_(pa.struct([
            ('name', pa.string()),
            ('quantity', pa.string()),
            ('purity', pa.string())
        ]))),
        ('steps', pa.list_(pa.struct([
            ('step_name', pa.string()),
            ('temperature', pa.string()),
            ('duration', pa.string()),
            ('description', pa.string())
        ]))),
        ('equipment', pa.list_(pa.struct([
            ('name', pa.string()),
            ('model', pa.string())
        ]))),
        ('notes', pa.string())
    ])),
    ('synthesis_extraction_performance_llm', pa.int32()),
    ('figure_extraction_performance_llm', pa.int32()),
    ('synthesis_extraction_performance_human', pa.int32()),
    ('figure_extraction_performance_human', pa.int32()),
    ('paper_title', pa.string()),
    ('paper_published_date', pa.string()),
    ('paper_abstract', pa.string()),
    ('paper_doi', pa.string()),
    ('paper_url', pa.string())
    ])