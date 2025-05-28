# Developer Guide
This file contains information for developers for this project

TL;DR
```
uv sync
uv pip install -e .
uv pip install <package-name> # install package without adding to pyproject.toml
uv add <package-name> # install and add package to project dependencies
```

## FAQs

<details>
<summary>What are API keys and why do I need them?</summary>
API keys are credentials used to authenticate with external services (e.g., LLM providers, OCR APIs). You store them in your .env file so our code can securely access these services without hardcoding secrets. **Under no circumstance should you share your API key, post or commit it anywhere!** You can get API keys on the websites of the respective providers.
</details>
<details>
<summary>What is uv sync and do I need it?</summary>
`uv` is a fantastic, light-weight and user-friendly dependency manager -- think of it as an alternative to `pip`. `uv sync` installs all dependencies defined in pyproject.toml into your virtual environment. Always run it after pulling changes to ensure you have the latest required packages.
</details>
<details>
<summary>My code doesn't compile and the issue is not resolved after consolidating ChatGPT for ~10 min. What do I do?</summary>
Please raise an issue in the GitHub repository with details about the error, steps to reproduce, and any relevant logs or screenshots (important for us to reproduce!). Our team will triage and help resolve it.
</details>
<details>
<summary>What is a pre-commit hook?</summary>
A pre-commit hook is a script that runs before each git commit. It ensures, for example, that the code adheres to python style guides. We use `uvx pre-commit install` to enforce formatting and linting automatically, catching issues early.
</details>
<details>
<summary>What is a lock file?</summary>
The lock file (`uv.lock`) pins exact versions of all dependencies. This ensures consistent installs across machines and CI runs. If `uv` causes issues for you, deleting the lock file and `.venv/` directory and reinstalling might do the job.
</details>
<details>
<summary>I still have questions or run into issues!</summary>
Get in touch with us -- preferably via the designated slack channel, feel free to also contact @mlederbauer on GitHub / Magdalena Lederbauer on slack.
</details>

## Structure of this repository

This repository contains
```
.
├── notebooks/                   # Jupyter notebooks for exploration and demos
├── scripts/                     # Utility scripts and one-off tools (cf. README.md)
├── src/                         # Python source code
│   └── llm_synthesis/           # Main package installed with uv pip install -e .
│       ├── extraction/          # Code for extracting content
│       │   ├── synthesis/       # Synthesis extraction modules
│       │   ├── figures/         # Image extraction modules
│       │   └── text/            # Synthesis paragraph extraction modules
│       ├── ontologies/          # Domain ontologies and schema definitions
│       └── utils/               # Helper functions and common utilities
├── developer_guide.md           # This developer guide
├── LICENSE                      # Project license (Apache 2.0)
├── pyproject.toml               # Project configuration (dependencies, metadata), managed by uv
├── README.md                    # High-level project overview and quickstart
└── uv.lock                      # Lockfile for uv package manager
```


## Installation

* Clone the repository
```
git clone https://github.com/LeMaterial/lematerial-synthesis-parser.git
cd lematerial-synthesis-parser
```
* Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
* Create a virtual environment (first time only):
```
uv venv -p 3.11 --seed
```
* Install dependencies with `uv`:
```
uv sync
```
* Install the LLM-Synthesis package:
```
uv pip install -e .
```
* Environment variables: Create a `.env` file in the project root with any required API keys:
```
MISTRAL_API_KEY=your_mistral_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```
* Install the pre-commit hook:
```
uvx pre-commit install
```

## Git Best Practices

### Branches
The main branch is reserved for the stable version of the code. When adding a new feature, we open a new *branch* and *merge* it with the main branch using a *pull request*:

```
git checkout main
git pull origin main
git checkout -b feat/new-feature-name
```

### Commit Messages

Make a habit of committing and pushing your code regularly (`git add <file-to-add> | git commit -m "commit message" | git push`)! Use the Conventional Commits style:

* feat: for new features
* fix: for bug fixes
* docs: for documentation changes
* style: for formatting, missing semicolons, etc.
* refactor: for refactoring code
* test: for adding or updating tests

Example:
```
❌ add markdown script
✅ feat(extraction): support image embedding in markdown output
```
### Pre-commit Formatting & Linting

Format and lint all Python files before committing:
```
uvx ruff format
uvx ruff check
```

## Submitting a Pull Request (PR)

Open a PR from your feature branch into `main`.
Link issues you are addressing (e.g., Closes #123).
Add a PR description:

```
Brief summary of changes

Any migration steps

Screenshots or examples if applicable
```
Respond to review comments promptly.
Squash & merge when approved.

## Documentation

After implementing any new feature, make sure to document it properly!
* README.md (if applicable): High-level overview and quickstart.
* Developer Guide (if applicable): (this file) for contributor onboarding.
* Docstrings (**mandatory, including type hints!**): ensure public APIs are documented.
* Notebooks (if applicable): examples and exploratory work under notebooks/.

## Contributing & Support

Feel free to open issues for bugs or feature requests.
For questions, don't hesitate to reach out to the maintainers:
* @mlederbauer (Magdalena Lederbauer via Slack)

Thank you for contributing to LeMaterial-Synthesis-Parser! 🎉