# Validation record

This edition validates the public implementation and archived evidence without redistributing course data or rerunning long training jobs.

## Completed locally

| Check | Result |
|---|---|
| Fresh Python 3.13 virtual environment; pinned CPU dependencies; `pip check` | Passed |
| Twelve focused integrity tests | Passed |
| Thirteen historical OOF prediction sets | Metrics and class reports consistent |
| Forty-eight historical evidence files | SHA-256 integrity checks passed |
| CNN fold summaries and pooled OOF JSON | Consistent with saved fold predictions |
| Synthetic CPU experiment and prediction export | Passed; temporary inputs and outputs removed |
| Four generated PNG figures | Inspected for readable labels, scales and annotations |
| Public allowlist and local Markdown links | Checked by `src.check_portfolio` |

The CPU installation was performed in a new virtual environment, rather than relying on the user's global packages. The complete resolved dependency list is recorded in [requirements-cpu-lock.txt](../requirements-cpu-lock.txt); the direct requirements remain in [requirements.txt](../requirements.txt). This records the portfolio validation environment, not the historical training environment.

The synthetic experiment checks data loading, shuffled feature joins, engineered features, stratified model selection, OOF predictions and the final `image_id,class_id` export. It uses no pretrained weights or raw course data. Synthetic scores are neither archived nor quoted as model performance.

Run the [reproduction commands](reproduction.md) to repeat these checks. The GitHub Actions workflow runs installation, integrity tests, result verification, the synthetic smoke command and the publication checks on Linux. Its live status is visible in the repository's Actions tab and README badge.

## Not performed

- No full original-data experiment, feature-cache regeneration, CNN training or GPU benchmark was run for this edition.
- No independent test labels, leaderboard evidence, statistical-significance test or pretraining-overlap audit was available.
- Historical input data and externally supplied feature preprocessing were not re-audited. Train/test non-overlap in archived CNN metadata remains an original recorded check, not a new inspection of the underlying data.
- Pinning packages now cannot recover an unrecorded historical training lock or guarantee identical CUDA results.

Successful evidence checks establish consistency between saved predictions and reported metrics. They do not remove the selection limitations described in [methodology](methodology.md).
