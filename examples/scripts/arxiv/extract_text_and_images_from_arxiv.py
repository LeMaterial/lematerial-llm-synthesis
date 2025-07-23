import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from arxiv_scraper import ArxivScraper
import time
import pandas as pd
from pathlib import Path
from schema import schema

def extract_text_and_files(destination="/fsx/georgia_channing/lemat_parquet/data/arxiv/arxiv_with_images.parquet"):
    
    dataset = load_dataset("LeMaterial/LeMat-Synth", data_files="data/arxiv/*.parquet", streaming=True)['train']
    batch_size = 10
    batch = []
    writer = None
    arxiv_scraper = ArxivScraper()
    i = 0
    total = 500
    
    metrics_dict = {}

    for example in dataset:
        i += 1
        time.sleep(3)
        text, images, method = arxiv_scraper.parse_from_id(example['id'])
        row = dict(example) 
        if text is None:
            row['pdf_extractor'] = f'failed {method} parsing'
        else:
            row['text_paper'] = text

        if images is not None:
            row['images'] = [
                {"path": path, "bytes": img_bytes}
                for path, img_bytes in images.items()
            ]

        metrics_dict[example['id']] = {'method':method, 'failed':True if text is None else False}

        # unrelated to this but need to guard to make run
        if 'structured_synthesis' not in row:
            row['structured_synthesis'] = None

        batch.append(row) 

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            table = table.cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(destination, schema)
            writer.write_table(table)
            batch = []
        
        if i > total:
            break
    
    if batch:
        table = pa.Table.from_pylist(batch)
        table = table.cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(destination, schema)
        writer.write_table(table)

    if writer:
        writer.close()

    return

if __name__=="__main__":
    extract_text_and_files()