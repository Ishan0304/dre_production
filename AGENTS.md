# Project Guidance

This project identifies patients who likely have drug-resistant epilepsy from longitudinal patient data.

`likely_dre` is a computable inference, not a formal adjudicated ILAE drug-resistant epilepsy diagnosis. Future modules must avoid overstating clinical claims and should present outputs as evidence-supported inference for review.

Core evidence handling must separate:

1. observed evidence
2. inferred evidence
3. missing evidence

Dataset insight reporting is a first-class subsystem. Profiling, missingness, temporal coverage, cohort composition, and related reporting should be treated as production deliverables, not notebook afterthoughts.

Notebook logic should stay thin and call package code. Core ingestion, normalization, feature extraction, definitions, modeling, and reporting logic belongs under `src`.

Avoid em dashes in prose.
