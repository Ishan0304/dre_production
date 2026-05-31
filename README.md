# dre_production

`dre_production` is a production-shaped Python repository for identifying patients who likely have drug-resistant epilepsy from longitudinal patient data.

The project is intended to support computable evidence review across large clinical datasets. It is not a formal adjudicated ILAE drug-resistant epilepsy diagnosis system.

## Clinical Framing

The target label is `likely_dre`. It represents a conservative computable inference from available records, not a definitive clinical diagnosis.

Future modules should label a patient `likely_dre` only when the record shows:

1. documented epilepsy or recurrent seizure-related care
2. evidence that at least two distinct antiseizure medication schedules were attempted
3. evidence of persistent seizure burden after escalation to the second ASM

The system must separate observed evidence, inferred evidence, and missing evidence. Absence of data should not be treated as proof that an event did or did not occur.

## Architecture Overview

The repository uses a modular `src` layout so clinical definitions, dataset handling, feature generation, modeling, and reporting can evolve independently.

Planned package areas:

- `core`: shared contracts, configuration helpers, and runtime utilities
- `ingestion`: source readers and large dataset loading interfaces
- `normalization`: source-specific mapping into common data contracts
- `features`: evidence and feature construction from normalized data
- `definitions`: computable label definitions and evidence rules
- `insights`: dataset insight reports, profiling, cohort summaries, and data quality checks
- `datasets`: dataset assembly, splitting, and reproducible manifests
- `modeling`: model training, evaluation, inference, and artifact persistence
- `reporting`: human-readable reports for cohorts, labels, model runs, and data quality

Notebooks are intended to remain thin orchestration and reporting layers. Durable logic should live in package code and be covered by tests.

## Repository Layout

```text
dre_production/
  configs/
  src/
    core/
    ingestion/
    normalization/
    features/
    definitions/
    insights/
    datasets/
    modeling/
    reporting/
  notebooks/
  reports/
  artifacts/
  data/
  tests/
```

## Installation

Dependencies are declared in `pyproject.toml`. A future development environment can be created with:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Tests

Tests will be run with:

```bash
pytest
```

Block 0 only creates the repository scaffold. Clinical business logic, notebooks, and model training code are intentionally not implemented yet.
