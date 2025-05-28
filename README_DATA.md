# Access to data

We use a bucket Google Cloud storage for accessing the PDFs, processed txt files, and results.

## Structure of the bucket

```
- pdf_files_eval: Includes a sub-selection of PDF papers for the benchmark evaluations.
- txt_files_eval: Includes a sub-selection of parsed papers for the benchmark evaluations.
- pdf_files_all: Includes all manually curated ammonia cracking papers (updated: May 22 2025)
- txt_files_all: Includes all manually curated ammonia cracking papers, parsed as txt (updated: May 22 2025)
```

## Use Google Cloud Storage

Make sure you are authenticated either by running `gcloud auth login` or by setting the environment variable `GOOGLE_APPLICATION_CREDENTIALS`

### Push data to GCS

If you want to synchronize a local folder on a distant folder:

```sh
gsutil -m rsync -r ./local-folder gs://your-bucket-name/target-folder
```

For example:

```sh
gsutil -m rsync -r ./data/pdf_papers gs://entalpic-prod-llm-synthesis-papers/test/pdf_papers
```

### Pull data from GCS

If you want to synchronize a distant folder on a local folder:

```sh
gsutil -m rsync -r gs://your-bucket-name/target-folder ./local-folder 
```

For example:

```sh
gsutil -m rsync -r  gs://entalpic-prod-llm-synthesis-papers/test/txt_papers ./data/txt_papers
```
