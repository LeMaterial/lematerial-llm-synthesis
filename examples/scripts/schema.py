import pyarrow as pa

schema = pa.schema([
    ('id', pa.string()),
    ('title', pa.string()),
    ('authors', pa.list_(pa.string())),
    ('abstract', pa.string()),
    ('doi', pa.string()),
    ('published_date', pa.string()),
    ('updated_date', pa.string()),
    ('categories', pa.string()),
    ('license', pa.string()),
    ('pdf_url', pa.string()),
    ('views_count', pa.int64()),
    ('read_count', pa.int64()),
    ('citation_count', pa.int64()),
    ('keywords', pa.list_(pa.string())),
    ('text_paper', pa.string()),
    ('text_si', pa.string()),
    ('source', pa.string()),
    ('pdf_extractor', pa.string()),
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
    ]))
    ])