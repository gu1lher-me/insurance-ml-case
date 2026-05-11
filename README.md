# Insurance ML Case

Resident-level incident-risk scoring for skilled nursing liability insurance.

This project builds a production-style risk pipeline for Tricura Insurance
Group. The goal is not only to predict clinical incidents, but to prioritize
the resident interventions most likely to reduce claim dollars.

## Business Challenge

Tricura provides liability insurance to skilled nursing facilities. A large
share of claim exposure comes from resident incidents such as falls, medication
errors, wounds, return-to-hospital events, elopement, and altercations.

The business lever is prevention. Raising premiums can hurt retention, so this
pipeline focuses on identifying resident-window risks early enough for facility
teams to act. The output is therefore framed in dollars:

- expected claim cost,
- expected avoidable claim cost,
- recommended action queue,
- captured claim exposure in backtesting,
- estimated avoided claim dollars, net savings, and ROI.

Average claim costs used by the policy layer:

| Incident type | Avg claim cost |
|---|---:|
| Falls | USD 3,500 |
| Medication errors | USD 5,000 |
| Wounds / pressure injuries | USD 4,000 |
| Return-to-hospital events | USD 20,000 |
| Elopement / wandering | USD 2,500 |
| Altercations | USD 2,500 |


## Installation

This project uses `uv` and Python 3.13.

```powershell
git clone <repo-url>
cd insurance-ml-case
uv sync
```

Run commands through the managed environment:

```powershell
uv run python --version
```

The data is expected under `data/raw/` as parquet files. The repository also
contains generated feature stores, model matrices, score outputs, and local
MLflow artifacts from the current run.


## What This Builds

The repository implements a tiered resident risk system:

| Tier | Incident types | Method | Why |
|---|---|---|---|
| Tier 1 | Falls, RTH, wounds | CatBoost temporal classifiers | Enough positive events for supervised prediction |
| Tier 2 | Altercations | Resident-level propensity model | Weekly labels are too sparse, but resident-level signal is learnable |
| Tier 3 | Medication errors, elopement | Point-in-time business rules | Events are too rare for stable supervised ML |
| Tier 4 | All supported risks | Composite expected-cost score | Converts heterogeneous risk signals into business priority |

The main scoring output is one row per resident-window with:

- event probabilities or rule flags,
- event-specific expected claim costs,
- total `composite_expected_cost`,
- `expected_avoidable_cost`,
- `economic_action`,
- reason codes and recommended actions for facility teams.

For the full modeling rationale, see
[`docs/modeling-strategy.md`](docs/modeling-strategy.md).

## Data Used

The raw anonymized backend tables live in `data/raw/`.

Tables used directly in the main feature pipelines:

| Table | Main use |
|---|---|
| `residents` | Resident spine, facility, age, length of stay |
| `vitals` | Recent vital-sign trends and monitoring indicators |
| `incidents` | Fall, wound, altercation history and target labels |
| `diagnoses` | Clinical risk flags and diagnosis counts |
| `hospital_transfers` | Return-to-hospital labels and prior RTH history |
| `hospital_admissions` | Active/recent admission context |
| `needs` | Open care-plan needs by category |
| `lab_reports` | Recent abnormal, critical, and total lab counts |
| `document_tags` | NLP-derived clinical risk tags |
| `medications` | Medication-error business-rule validation |

Some raw tables are intentionally excluded from the model features because
coverage is too sparse for this proof of concept, including `adl_responses`,
`gg_responses`, and `therapy_tracks`. See
[`docs/feature-store-docs.md`](docs/feature-store-docs.md) for details.

## Feature Sets

The 7-day and 14-day supervised matrices use the same 223 engineered features.
The feature groups are:

| Feature group | Description |
|---|---|
| Demographics | Age at window and length of stay |
| Vitals | Rolling statistics and measurement flags across 3, 7, and 14 day lookbacks |
| Diagnoses | Active diagnosis counts, ICD chapter counts, and fall/RTH risk flags |
| Incident history | Prior falls, wounds, and altercations over recent and all-time lookbacks |
| RTH history | Prior unplanned hospital transfer counts |
| Admission context | Active admission status and recent hospital-stay context |
| Care-plan needs | Open need counts by care category |
| Labs | Recent abnormal, critical, and total lab counts |
| Document tags | Binary clinical tag indicators such as fall risk, wound risk, dementia, pain, infection, wandering, and elopement |

Saved feature artifacts:

| Artifact | Rows | Contents |
|---|---:|---|
| `data/feature_store/features_7d.parquet` | 66,312 | Target-free 7-day feature store |
| `data/processed/model_matrix_7d.parquet` | 66,312 | 7-day features plus `fall_7d` and `rth_7d` |
| `data/feature_store/features_14d.parquet` | 32,361 | Target-free 14-day feature store |
| `data/processed/model_matrix_14d.parquet` | 32,361 | 14-day features plus `wound_14d` |

## Chronological Design

The modeling unit is a resident-window. A resident is scored at `window_start`
using only information available before that date.

Historical matrices were built over the signal window `2023-07-01` to
`2025-02-01`. The falls/RTH matrix uses non-overlapping 7-day windows, while
the wound matrix uses non-overlapping 14-day windows.

Target horizons:

| Target | Horizon | Label definition |
|---|---:|---|
| `fall_7d` | 7 days | Any fall in `[window_start, window_end)` |
| `rth_7d` | 7 days | Any unplanned hospital transfer in `[window_start, window_end)` |
| `wound_14d` | 14 days | Any wound in `[window_start, window_end)` |

Leakage controls:

- `feature_cutoff = window_start - 1 day`
- features use source records known at or before `feature_cutoff`
- hospital admission context checks both event dates and row creation dates
- target windows are separated from feature lookback windows

Train/test/holdout splitting is chronological:

| Split | Window rule | Purpose |
|---|---|---|
| Train | `window_start < 2024-07-01` | Fit candidate models |
| Test | `2024-07-01 <= window_start < 2025-01-01` | Temporal validation and calibration checks |
| Holdout | `window_start >= 2025-01-01` | Simulated first production month |

The training code also uses expanding-window temporal validation. Fold logic
purges training labels whose outcome window would cross into the test period.
That matters because a row with `window_start` before a boundary can still have
an outcome window that reaches after the boundary.

## Modeling Strategy

Tier 1 models estimate near-term probabilities:

- `P(fall in next 7 days)`
- `P(return to hospital in next 7 days)`
- `P(wound in next 14 days)`

These are CatBoost binary classifiers because the feature matrix has meaningful
missingness and a mix of nonlinear clinical signals.

The altercation model is resident-level rather than weekly. Its output is
scaled to the observed weekly base rate before entering the composite score.

Medication-error and elopement risks use rule flags. Their expected-cost
contribution is based on retrospective positive predictive value of the rule
sets.

The composite score is:

```text
composite_expected_cost =
    fall_probability * 3500
  + rth_probability * 20000
  + wound_probability * 4000
  + altercation_probability * 2500
  + med_error_probability * 5000
  + elopement_probability * 2500
```

Expected avoidable cost then applies pilot intervention-effectiveness
assumptions from `modeling/business_policy.py`:

| Incident type | Assumed effectiveness |
|---|---:|
| Falls | 20% |
| RTH | 15% |
| Wounds | 20% |
| Altercations | 15% |
| Medication errors | 20% |
| Elopement | 25% |

The default economic action rule is:

```text
economic_action = expected_avoidable_cost >= alert_review_cost
```

The default alert review cost is USD 100.

## Backtest Results

The financial backtest uses January 2025 as a simulated production month.

Backtest setup:

| Item | Value |
|---|---:|
| Scoring period | 2025-01-01 to 2025-02-01 |
| Scored resident-windows | 3,612 |
| Actual modeled claim events | 253 |
| Actual modeled claim exposure | USD 1,788,500 |
| Alert review cost | USD 100 |
| Primary policy | Top 10% per facility |

Primary policy result, `top_10pct_per_facility`:

| Metric | Value |
|---|---:|
| Alerts | 401 |
| Alert rate | 11.1% |
| Captured claim exposure | USD 304,500 |
| Claim exposure capture rate | 17.0% |
| Estimated avoided claim dollars | USD 53,900 |
| Intervention cost | USD 40,100 |
| Estimated net savings | USD 13,800 |
| Estimated ROI | 0.34x |

Policy sensitivity:

| Policy | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Top 5% per facility | 226 | USD 192,000 | USD 34,400 | USD 22,600 | USD 11,800 | 0.52x |
| Top 10% per facility | 401 | USD 304,500 | USD 53,900 | USD 40,100 | USD 13,800 | 0.34x |
| Top 15% per facility | 590 | USD 449,500 | USD 79,775 | USD 59,000 | USD 20,775 | 0.35x |
| Top 20% per facility | 753 | USD 584,500 | USD 103,775 | USD 75,300 | USD 28,475 | 0.38x |
| Economic threshold | 423 | USD 320,000 | USD 58,875 | USD 42,300 | USD 16,575 | 0.39x |

The most important metric is captured claim exposure: dollars from actual
holdout events that had an alert before the event occurred. Net savings and ROI
are decision metrics, but they depend on the assumed intervention effectiveness
and review cost. They should be validated in a prospective pilot.

For the generated report, see
[`docs/backtest-results.md`](docs/backtest-results.md).

## Reproducing the Pipeline

Build historical 7-day features and labels:

```powershell
uv run python feature_engineering\build_falls_rth_matrix.py `
  --signal-start 2023-07-01 `
  --signal-end 2025-02-01 `
  --feature-out data\feature_store\features_7d.parquet `
  --model-out data\processed\model_matrix_7d.parquet
```

Build historical 14-day wound features and labels:

```powershell
uv run python feature_engineering\build_wound_matrix.py `
  --signal-start 2023-07-01 `
  --signal-end 2025-02-01 `
  --feature-out data\feature_store\features_14d.parquet `
  --model-out data\processed\model_matrix_14d.parquet
```

Train final Tier 1 models:

```powershell
uv run python modeling\train_models.py
```

Optional hyperparameter tuning:

```powershell
uv run python modeling\train_models.py `
  --tune-hyperparameters `
  --hyperparameter-trials 10
```

Train the altercation propensity model:

```powershell
uv run python modeling\train_altercation.py
```

Score the January 2025 holdout:

```powershell
uv run python modeling\score_composite.py `
  --holdout-start 2025-01-01 `
  --output data\scored\composite_scores_holdout.parquet `
  --model-source mlflow `
  --model-phase final_model
```

Run the financial backtest:

```powershell
uv run python modeling\backtest_financials.py `
  --scores-path data\scored\composite_scores_holdout.parquet `
  --alert-cost 100
```

## Daily Scoring

Score a new as-of date without retraining:

```powershell
uv run python pipelines\run_daily_scoring.py `
  --as-of-date 2025-02-01 `
  --model-phase final_model `
  --action-policy economic
```

This writes:

| Artifact | Description |
|---|---|
| `data/feature_store/scoring/features_YYYY-MM-DD.parquet` | Target-free features for active residents |
| `data/scored/composite_scores_YYYY-MM-DD.parquet` | Full resident score table |
| `data/scored/composite_scores_YYYY-MM-DD_top.csv` | Top-score preview |
| `data/action_queues/action_queue_YYYY-MM-DD.csv` | Operational action queue |

Retrain latest production models, then score:

```powershell
uv run python pipelines\run_daily_scoring.py `
  --as-of-date 2025-02-01 `
  --label-cutoff 2025-02-01 `
  --signal-start 2023-07-01 `
  --rebuild-training-features `
  --retrain `
  --action-policy economic
```

Daily scoring parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `--as-of-date` | Required scoring date | none |
| `--label-cutoff` | Latest fully labeled training date | `--as-of-date` |
| `--signal-start` | Historical start date when rebuilding features | `2023-07-01` |
| `--rebuild-training-features` | Rebuild historical 7-day and 14-day matrices | off |
| `--retrain` | Train latest production MLflow models before scoring | off |
| `--model-phase` | MLflow phase to load | `production_model` if retraining, else `final_model` |
| `--action-policy` | Select `economic` or `capacity` action queue | `economic` |
| `--capacity-rate` | Facility-level action rate for capacity policy | `0.10` |

## Repository Layout

| Path | Purpose |
|---|---|
| `docs/problem-description.md` | Original take-home assignment context |
| `docs/modeling-strategy.md` | Detailed modeling and financial methodology |
| `docs/backtest-results.md` | Generated financial backtest summary |
| `docs/feature-store-docs.md` | Feature-store artifacts and temporal design |
| `docs/prediction-pipeline.md` | New-batch and daily-scoring operations guide |
| `feature_engineering/` | Historical and target-free feature builders |
| `modeling/` | Training, scoring, rules, policy, and backtesting code |
| `pipelines/run_daily_scoring.py` | End-to-end daily scoring workflow |
| `data/raw/` | Anonymized source tables |
| `data/processed/` | Model-ready matrices |
| `data/feature_store/` | Target-free feature stores |
| `data/scored/` | Composite score outputs |
| `data/action_queues/` | Operational action queues |

## Interpretation

This backtest shows that the system can put meaningful claim dollars in front
of facility teams before incidents occur. The January 2025 primary policy
captured USD 304,500 of claim exposure with 401 alerts, producing an estimated
USD 13,800 in net savings under the pilot assumptions.

Those savings are not causal proof. The recommended next step is a prospective
pilot that measures actual claims versus expected claims after deployment,
while controlling for facility baseline risk and resident mix.
