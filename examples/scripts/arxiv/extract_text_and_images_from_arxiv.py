import gc
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from arxiv_scraper import ArxivScraper
from datasets import load_dataset
from schema import schema


def compute_method_stats(data):
    method_counts = defaultdict(int)
    fail_counts = defaultdict(int)

    for entry in data.values():
        method = entry["method"]
        failed = entry["failed"]

        method_counts[method] += 1
        if failed:
            fail_counts[method] += 1

    total = sum(method_counts.values())
    stats = {}

    for method in method_counts:
        count = method_counts[method]
        fails = fail_counts[method]
        stats[method] = {
            "percentage": round((count / total) * 100, 2),
            "fail_rate": round((fails / count) * 100, 2),
            "count": count,
            "fails": fails,
        }

    return stats


def extract_text_and_files(
    destination="/fsx/georgia_channing/lemat_parquet/data/arxiv/arxiv_with_images.parquet",
):
    dataset = load_dataset(
        "LeMaterial/LeMat-Synth",
        data_files="data/arxiv/*.parquet",
        streaming=True,
    )["train"]
    batch_size = 100
    batch = []
    writer = None
    arxiv_scraper = ArxivScraper(temp_dir="/fsx/georgia_channing/temp")

    metrics_dict = {}

    for example in dataset:
        time.sleep(1)
        text, images, method = arxiv_scraper.parse_from_id(example["id"])
        row = dict(example)
        if text is None:
            row["pdf_extractor"] = f"failed {method} parsing"
        else:
            row["text_paper"] = text

        if images is not None:
            row["images"] = [
                {"path": path, "bytes": img_bytes}
                for path, img_bytes in images.items()
            ]

        metrics_dict[example["id"]] = {
            "method": method,
            "failed": True if text is None else False,
        }

        # unrelated to this but need to guard to make run
        if "structured_synthesis" not in row:
            row["structured_synthesis"] = None

        batch.append(row)

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            table = table.cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(destination, schema)
            writer.write_table(table)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch)
        table = table.cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(destination, schema)
        writer.write_table(table)

    if writer:
        writer.close()

    print(compute_method_stats(metrics_dict))

    return


if __name__ == "__main__":
    extract_text_and_files()
    gc.collect()
