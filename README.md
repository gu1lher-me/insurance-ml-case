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

### Altercation Model In Practice

Altercations are modeled differently from falls, RTH, and wounds because the
weekly event rate is too low for a reliable resident-window classifier. In the
current data there are 257 altercation events across 127 residents, about a
4.2% resident-level positive rate, but the 7-day resident-window event rate is
only about 0.24%. A weekly classifier would mostly learn that almost every
window is negative and would produce unstable tail probabilities.

The altercation model therefore answers a different question:

```text
Is this resident behaviorally prone to altercation during their stay?
```

The target is `has_altercation`, a resident-level binary label equal to 1 when
the resident has any altercation incident in the training data available before
the scoring cutoff. The model is a CatBoost classifier trained on resident-level
features rather than window-level features.

Altercation feature groups:

| Feature group | Examples |
|---|---|
| Demographics and stay context | Age, length of stay |
| Diagnosis flags | Dementia, Alzheimer's, schizophrenia, schizoaffective disorder, bipolar disorder, depression, anxiety, alcohol-related diagnoses |
| Care-plan needs | Psychotropic, cognitive, behavioral, and mood-related needs |
| Document tags | Aggressive behavior, dementia, psychotropic medications, hallucinations/delusions, mental status, anxiety, depression, impaired mobility |
| Non-altercation incident history | Prior falls, wounds, and total prior non-altercation incidents |

Prior altercation counts are intentionally excluded from the feature set to
avoid a trivial leakage feature. The model can still learn clinically useful
behavioral and cognitive risk patterns from diagnoses, care needs, notes-derived
tags, demographics, and other incident history.

The current resident-level model is used as a propensity ranker. Cross-validated
performance is strong for this purpose:

| Metric | Value |
|---|---:|
| CV ROC-AUC | 0.8837 |
| CV PR-AUC | 0.2995 |

At scoring time, the pipeline does not use the raw resident propensity directly
as a 7-day event probability. A resident-level "ever altercation" probability
is much larger than the chance of an altercation in the next week. To make the
score financially comparable with the other event models, the scorer rescales
the resident propensity to the observed historical weekly base rate:

```text
altercation_probability =
    altercation_resident_probability
  * historical_weekly_altercation_base_rate
  / mean_resident_propensity
```

The value is clipped to `[0, 1]`. The output keeps both numbers:

| Output column | Meaning |
|---|---|
| `altercation_resident_probability` | Resident-level propensity score |
| `altercation_probability` | Weekly probability used in expected-cost scoring |
| `altercation_weekly_base_rate` | Historical 7-day base rate used for scaling |
| `altercation_probability_scale` | Multiplicative scale applied to resident propensity |

The weekly probability then contributes to the composite score:

```text
altercation_expected_cost = altercation_probability * 2500
altercation_expected_avoidable_cost = altercation_expected_cost * 0.20
```

In the action queue, altercation is handled as one possible reason for review.
If it is one of the top expected-cost contributors, the resident receives
`model:altercation` in `reason_codes`, `altercation` as `top_reason` when it is
the largest contributor, and this suggested action:

```text
Behavioral-care review: evaluate triggers, psychotropic monitoring, staffing plan, and resident interactions.
```

Operationally, this means the altercation model is not meant to say "this
resident will have an altercation this week" with high certainty. It is meant
to surface residents whose behavioral profile makes altercation prevention
worth considering, then price that signal conservatively as a weekly expected
claim-cost contribution.

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

Expected avoidable cost applies one uniform intervention-effectiveness
scenario from `modeling/business_policy.py`:

```text
expected_avoidable_cost = composite_expected_cost * 0.20
```

This 20% baseline is not a learned causal estimate. It is a simple POC
assumption chosen to avoid false precision across incident types. The backtest
also reports sensitivity at 10%, 15%, 20%, and 25%; incident-specific
effectiveness should only be added after a prospective pilot measures it.

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
| Estimated avoided claim dollars | USD 60,900 |
| Intervention cost | USD 40,100 |
| Estimated net savings | USD 20,800 |
| Estimated ROI | 0.52x |

Policy sensitivity:

| Policy | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Top 5% per facility | 226 | USD 192,000 | USD 38,400 | USD 22,600 | USD 15,800 | 0.70x |
| Top 10% per facility | 401 | USD 304,500 | USD 60,900 | USD 40,100 | USD 20,800 | 0.52x |
| Top 15% per facility | 590 | USD 449,500 | USD 89,900 | USD 59,000 | USD 30,900 | 0.52x |
| Top 20% per facility | 753 | USD 584,500 | USD 116,900 | USD 75,300 | USD 41,600 | 0.55x |
| Economic threshold | 455 | USD 334,000 | USD 66,800 | USD 45,500 | USD 21,300 | 0.47x |

Primary policy effectiveness sensitivity:

| Assumed effectiveness | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 401 | USD 304,500 | USD 30,450 | USD 40,100 | -USD 9,650 | -0.24x |
| 15% | 401 | USD 304,500 | USD 45,675 | USD 40,100 | USD 5,575 | 0.14x |
| 20% | 401 | USD 304,500 | USD 60,900 | USD 40,100 | USD 20,800 | 0.52x |
| 25% | 401 | USD 304,500 | USD 76,125 | USD 40,100 | USD 36,025 | 0.90x |

The most important metric is captured claim exposure: dollars from actual
holdout events that had an alert before the event occurred. Net savings and ROI
are decision metrics, but they depend on the uniform effectiveness scenario and
review cost. They should be validated in a prospective pilot.

For the generated report, see
[`docs/backtest-results.md`](docs/backtest-results.md).

## Next Steps for Production

The current system is validated in backtest. To move to production, consider:

### A/B Testing & Causal Validation
- Deploy model predictions to a subset of facilities while maintaining control groups
- Compare alerts + interventions vs. baseline over 2–3 months
- Measure actual incidents, claims, and facility engagement rates
- Use difference-in-differences or causal forest to account for facility baseline risk and resident mix
- Decision: roll out to full portfolio or refine model based on pilot results

### Cloud Deployment
- Containerize the training and scoring pipelines (Docker)
- Deploy to AWS (S3 for data/artifacts, EC2/SageMaker/K8s pods for compute)
- Move MLflow tracking server to remote instance and store artifacts on S3

### Orchestration & Workflow
- Build Airflow DAGs for:
  - **Daily feature engineering**: compute 7d and 14d features for active residents
  - **Drift monitoring**: detect feature/label/prediction drift and alert on performance degradation
  - **Model retraining**: retrain Tier 1 and altercation models using expanding-window logic
  - **Batch scoring**: score all residents and generate action queues  
- Integrate with MLflow model registry for governance (staging → production promotion)
- Add data validation and error-handling; configure alerting and logging

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
  --alert-cost 100 `
  --intervention-effectiveness 0.20 `
  --effectiveness-sensitivity 0.10 0.15 0.20 0.25
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
USD 20,800 in net savings under the 20% baseline effectiveness scenario.

Those savings are not causal proof. The recommended next step is a prospective
pilot that measures actual claims versus expected claims after deployment,
while controlling for facility baseline risk and resident mix.
