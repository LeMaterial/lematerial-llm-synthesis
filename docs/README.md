# Documentation — Contributor Guide

This file is for contributors working on the docs. It is intentionally not
listed in `mkdocs.yml`'s `nav` and will not appear in the generated site.

## Prerequisites

Docs dependencies live in the `dev` group:

```bash
uv sync --group dev
```

## Local development

Start a live-reloading server — the browser auto-refreshes on every file save:

```bash
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000>.

## Build (static HTML)

```bash
uv run mkdocs build        # output lands in site/
```

`site/` is git-ignored. Run `mkdocs build --strict` to treat warnings
(including pages not in `nav`) as errors.

## API reference pages

Pages under `docs/api/` pull docstrings from source via `mkdocstrings`.
After editing a docstring, a running `mkdocs serve` picks up the change
automatically. No separate step needed.

## Making the docs publicly accessible

### Option A — GitHub Pages (recommended, zero infra)

One-shot deploy from your local machine to the `gh-pages` branch:

```bash
uv run mkdocs gh-deploy
```

Then enable Pages on GitHub: **Settings → Pages → Branch: `gh-pages` / `/ (root)`**.
The live URL will be `https://<org>.github.io/lematerial-llm-synthesis/`.

For the canonical repo this is already configured in `mkdocs.yml`:

```
site_url: https://lematerial.github.io/lematerial-llm-synthesis/
```

### Option B — GitHub Actions (auto-deploy on push)

Add `.github/workflows/docs.yml`:

```yaml
name: Deploy docs
on:
  push:
    branches: [main]
    paths: ["docs/**", "mkdocs.yml"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install uv && uv sync --group dev
      - run: uv run mkdocs gh-deploy --force
```

### Option C — Read the Docs

Connect the repo at <https://readthedocs.org>. Add a `.readthedocs.yaml`
at the repo root pointing at `mkdocs.yml`. RTD handles builds and hosting
for free on public repos.
