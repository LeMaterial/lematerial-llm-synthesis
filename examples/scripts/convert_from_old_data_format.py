from datasets import load_dataset, Dataset
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import feedparser
import time
from schema import schema

lemat = 'LeMaterial/LeMat-Synth'

def convert_splits(splits=['arxiv', 'chemrxiv', 'omg24']):
    for split in splits:
        to_parquet(split=split, destination=f"/fsx/georgia_channing/lemat_parquet/data/{split}/{split}.parquet")
    return

def to_parquet(split=None, datafiles=None, destination=None, batch_size=None):
    if datafiles:
        streamed_dataset = load_dataset(lemat, data_files=datafiles, streaming=True)['train']
    elif split:
        streamed_dataset = load_dataset(lemat, data_files=datafiles, streaming=True)
    else:
        raise ValueError("Need either split or datafiles.")
    
    if batch_size == None:
        batch_size = 100
    writer = None

    batch = []
    i = 0
    for example in streamed_dataset:
        i += 1
        if 'images' not in example.keys():
            example["images"] = None
        if "structured_synthesis" not in example.keys():
            example["structured_synthesis"] = None
        if example['keywords'] is not None:
            example['keywords'] = list(example['keywords'])
        batch.append(example)
        
        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            table = table.cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(destination, schema)
            writer.write_table(table)
            print(f"Writing batch at i={i} (processed {i+1} rows)")
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch)
        table = table.cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(destination, schema)
        writer.write_table(table)

    if writer:
        writer.close()

    return

to_parquet(datafiles='data/chemrxiv-*.parquet', split='chemrxiv', destination=f"/fsx/georgia_channing/lemat_parquet/data/chemrxiv_filtered/chemrxiv.parquet")

def create_new_branch():
    from huggingface_hub import HfApi
    api = HfApi()


    refs = api.list_repo_refs(lemat, repo_type="dataset")
    new_branch = 'v'+str(len(refs.branches)+1)

    api.create_branch(
        repo_id=lemat,
        branch=new_branch,
        repo_type="dataset"
    )
    print("new branch is: ", new_branch)
    return new_branch


def get_urls_from_ids_with_api(ids):
    time.sleep(3)
    urls = []
    base_url = "http://export.arxiv.org/api/query"
    query = "id_list=" + ",".join(ids)
    response = requests.get(f"{base_url}?{query}&max_results=100")
    feed = feedparser.parse(response.text)
    for entry in feed.entries:
        for l in entry.links:
            if l.type == "application/pdf":            
                urls.append(l.href)

    return urls


def add_arxiv_urls():
    lemat = 'LeMaterial/LeMat-Synth'
    split = 'arxiv'

    streamed_dataset = load_dataset(lemat, split=split, streaming=True)
    parquet_file = f"/fsx/georgia_channing/lemat_parquet/data/{split}/{split}.parquet"
    batch_size = 100
    writer = None

    ids = []
    batch = []
    i = 0
    for example in streamed_dataset:
        i += 1
        ids.append(example['id'])
        batch.append(example)
        
        if len(batch) >= batch_size:
            urls = get_urls_from_ids_with_api(ids)
            if len(urls) != len(batch):
                raise ValueError(f"URL count {len(urls)} != batch size {len(batch)}")
            
            table = pa.Table.from_pylist(batch)

            url_array = pa.array(urls, type=pa.string())
            col_index = table.schema.get_field_index("pdf_url")
            if col_index == -1:
                raise ValueError("Column 'pdf_url' does not exist in the table schema")

            table = table.set_column(col_index, "pdf_url", url_array)
            table = table.cast(schema)

            if writer is None:
                writer = pq.ParquetWriter(parquet_file, schema)
            writer.write_table(table)
            print(f"Writing batch at i={i} (processed {i+1} rows)")
            ids = []
            batch = []
            urls = []
    

    if batch:
        urls = get_urls_from_ids_with_api(ids)
        if len(urls) != len(batch):
            raise ValueError(f"URL count {len(urls)} != batch size {len(batch)}")
        
        table = pa.Table.from_pylist(batch)

        url_array = pa.array(urls, type=pa.string())
        col_index = table.schema.get_field_index("pdf_url")
        if col_index == -1:
            raise ValueError("Column 'pdf_url' does not exist in the table schema")

        table = table.set_column(col_index, "pdf_url", url_array)
        table = table.cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, schema)
        writer.write_table(table)

    if writer:
        writer.close()
    return