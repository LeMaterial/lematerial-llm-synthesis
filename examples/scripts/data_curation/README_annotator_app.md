# Human Annotation App

Use the Streamlit app to complete human recipes and score extractor outputs.

## Prerequisites

- Create and seed the project venv:
  - `uv venv -p 3.11 --seed`
- Install project deps with `uv`:
  - `uv sync && uv pip install -e .`
- If `uv sync` fails on your platform, install Streamlit directly (pinned to validated version):
  - `pip install "streamlit==1.55.0"`
- Run commands from repository root so `annotations/` resolves correctly.

## Run the App

Canonical location:

```bash
streamlit run examples/scripts/data_curation/annotator_app.py
```

Backward-compatible entrypoint:

```bash
streamlit run annotator_app.py
```

## In-App Workflow

1. Select a paper ID.
2. Open/read the PDF in the app.
3. Fill or update `human_recipe`.
4. Score each extractor tab.
5. Save to `annotations/<paper_id>/result_human.json`.

## Submitting Annotations

After saving in the app:

1. Confirm updated files in `annotations/<paper_id>/`.
2. Stage only relevant files (avoid unrelated annotation changes).
3. Commit and push to the branch backing your PR.

Example:

```bash
git add annotations/<paper_id>/result_human.json
git commit -m "annotate/<paper_id>"
git push
```

Optional (create a dedicated branch and open a PR):

```bash
git fetch origin
git checkout -b annotate/<paper_id> origin/main
git add annotations/<paper_id>/result_human.json
git commit -m "annotate/<paper_id>"
git push -u origin annotate/<paper_id>
git log --oneline origin/main..HEAD
gh pr create --fill
```
