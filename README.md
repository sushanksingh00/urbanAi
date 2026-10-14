# UrbanAI — UrbanFlow AI

UrbanAI is a research and development project focused on intelligent traffic forecasting using spatio-temporal graph neural networks (STGNNs). The project explores how historical traffic measurements and relationships between road sensors can be used to predict near-future traffic conditions.

> **Project status:** Academic course project under development. Results and implementation details should be interpreted in the context of the experiments documented in this repository.

## Project goals

- Prepare and inspect traffic time-series data for forecasting.
- Build and evaluate baseline and graph-based forecasting models.
- Study multi-scale temporal patterns in traffic data.
- Analyze prediction errors and model uncertainty.
- Test model behavior under simulated abnormal or noisy conditions.
- Explore context-aware modeling and post-hoc explainability.

## Repository structure

```text
urbanAI/
├── data/             # Data directory
├── docs/             # Project documentation and supporting notes
├── experiments/      # Experiment scripts, outputs, or experiment-related files
├── notebooks/        # Research, preprocessing, training, and analysis notebooks
├── src/              # Reusable project source code
├── urbanAi-data/     # Additional saved project data
├── .env.example      # Example environment-variable configuration
├── .gitignore        # Git ignore rules
├── latest_plan.md    # Current project plan
├── old_plan.md       # Earlier project plan
├── PLAN.md           # Project plan
├── plann.md          # Additional planning notes
├── project_paths.py  # Project path configuration/helper
├── rem.md            # Project reminders or notes
└── requirements.txt  # Python dependencies
```

Folder contents may evolve as the project develops. Refer to the files in each directory for the authoritative details of what is currently stored there.

## Research workflow

The repository is organized around the following workflow:

1. **Explore the data** — inspect the available traffic measurements and their structure.
2. **Preprocess** — prepare chronologically ordered data, scale using training data, and create input/output windows.
3. **Establish baselines** — compare simple forecasting approaches with neural models.
4. **Train graph-based models** — investigate STGCN and multi-scale STGCN approaches.
5. **Evaluate** — calculate forecasting metrics and inspect errors across prediction horizons.
6. **Run additional analyses** — examine uncertainty, stress-test behavior, context features, and explainability where implemented.

The exact experiments and results should be taken from the executed notebooks and experiment records, rather than inferred from this overview.

## Dataset and saved data

The repository contains both `data/` and `urbanAi-data/`. Their exact contents should be verified by inspecting those directories.

Before committing data, check dataset licenses, file sizes, and whether the files contain credentials or other sensitive information. Large generated arrays, model checkpoints, and intermediate outputs are often better stored in cloud storage or tracked with Git LFS when appropriate, rather than committed directly to ordinary Git history.

**Do not assume that all local data is included in GitHub.** Check the repository's tracked files and `.gitignore` rules before sharing the project.

## Setup

Use Python in a virtual environment. From the repository root:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

Install the listed dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency fails to install, check the Python version and the package's compatibility requirements. The exact supported Python version is not specified in this README.

## Running the project

Start by reviewing the notebooks in `notebooks/` in their existing order and read relevant documentation in `docs/`. Some notebooks may depend on prepared datasets, saved model files, or paths specific to the original runtime.

- Preserve the existing executed notebooks and their outputs when sharing the reviewed work.
- Check notebook cells for local paths before running them in a new environment.
- Review `project_paths.py` and any configuration files before changing paths.
- Do not rerun notebooks solely to prepare the repository for sharing; rerunning can overwrite outputs or change recorded results.

Because the exact entry-point scripts and data availability are not specified here, there is no single guaranteed command to run the entire project end-to-end.

## Evaluation and reporting

Use the experiment notebooks and saved result files as the source of truth for reported metrics. When comparing experiments, ensure that results come from compatible runs, configurations, preprocessing steps, and data splits. Do not combine numbers from separate runs unless their comparability has been verified.

The research paper and supporting results should be kept synchronized with the experiments that are actually being reported.

## Reproducibility notes

For more reliable reproduction:

- Keep data splits chronological for time-series forecasting.
- Fit preprocessing transformations on the training split only, then apply the same transformation to validation and test data.
- Record model configuration, random seeds, dataset version, and evaluation settings.
- Keep large data artifacts separate from source code when practical.
- Document any environment-specific paths or unavailable external files.

These are project recommendations; consult the actual implementation to confirm which are already enforced.

## Security and repository hygiene

- Do not commit `.venv/`, `__pycache__/`, credentials, access tokens, or private environment files.
- Keep `.env.example` limited to placeholder values.
- Check `.gitignore` before adding datasets or model artifacts.
- Avoid deleting or editing executed notebooks during cleanup unless you have a separate backup and explicitly intend to change them.

## Current limitations

This README is a high-level guide based on the visible repository layout. It does not claim that every experiment is runnable from a fresh clone, that all data is tracked, or that every analysis has been fully validated. Confirm those details from the actual code, notebook outputs, and documentation.

## License

No license is specified in this README. Add a `LICENSE` file before presenting the repository as open source, and make sure any dataset redistribution is permitted by its terms.
