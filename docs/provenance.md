# Provenance and publication boundary

## Authorship and origins

The account owner, `tqyyds`, identifies this as an independently completed COMP30027 coursework project. This repository is a portfolio adaptation of the `_verify_comp30027_v7` archive, preserving the original modelling approach and saved experimental evidence. It is not an official university repository or an endorsement by the course.

The adaptation adds English documentation, reproducible figures, evidence verification, synthetic checks, integrity tests, CI and safe public packaging. Minimal implementation fixes make PyTorch optional for classical feature loading, reject malformed metadata/feature IDs, validate class mappings, and prevent automatic selection from smoke-only results. PowerShell wrappers use the recorded task-specific fold counts and default to synthetic smoke inputs when no data path is supplied. Historical predictions and reports are unmodified.

## Evidence hierarchy

1. Saved OOF predictions supply the recomputed headline metrics and figures.
2. Saved class reports, result-table rows and CNN metric JSON/fold summaries are consistency checks against those predictions.
3. Classical rows without an included corresponding OOF file remain `summary_only`; no prediction-level reproduction is claimed for them.
4. Earlier prose-only Kaggle feedback is not used as verified evidence.

`results/manifest.json` records each experiment's configuration, class mapping, evidence paths and the SHA-256 digest of every included historical CSV/JSON. These hashes detect changes to archived files relative to this manifest; they do not authenticate external data sources or establish independent generalization.

The original comparison table is preserved byte-for-byte in `results/historical/`. The presentation table adds an evidence label and removes only fully identical rows; all 29 supplied rows remain because they are distinct. Animal and bird prediction IDs/true labels are cross-checked across methods.

## Included and excluded materials

Included: implementation source, selected historical result tables, prediction IDs/class labels, per-class reports, experiment metadata, aggregate figures and documentation. The publication allowlist is [public-files.json](../public-files.json).

Excluded: raw images and course feature datasets, assignment instructions, the separately submitted report, raw-image montages, trained weights, feature caches, environments, credentials, absolute local paths and transient prediction submissions. The original archive remains separate from this portfolio copy.

Course-provided data/features and third-party pretrained weights retain their respective ownership and usage terms. Public visibility does not by itself grant a licence to reuse those materials. No repository-wide open-source licence is added in this edition.
