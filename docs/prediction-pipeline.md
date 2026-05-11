# New-Batch Prediction Pipeline

This project now has a production-style daily scoring workflow for new raw data
batches.

## Daily Scoring Without Retraining

Use existing MLflow model artifacts and build a fresh target-free scoring matrix:

```powershell
uv run python pipelines\run_daily_scoring.py --as-of-date 2025-02-01
```

Outputs:

| Artifact | Description |
|---|---|
| `data/feature_store/scoring/features_YYYY-MM-DD.parquet` | One target-free feature row per active resident |
| `data/scored/composite_scores_YYYY-MM-DD.parquet` | Full scored resident table |
| `data/scored/composite_scores_YYYY-MM-DD_top.csv` | Top 50 score preview |
| `data/action_queues/action_queue_YYYY-MM-DD.csv` | Resident action queue with suggested interventions |

## Daily Scoring With Latest Model Training

If a new batch includes enough newly matured labels, rebuild historical features,
train latest production models, and score the new batch:

```powershell
uv run python pipelines\run_daily_scoring.py `
  --as-of-date 2025-02-01 `
  --label-cutoff 2025-02-01 `
  --rebuild-training-features `
  --retrain
```

This performs:

1. rebuild 7-day falls/RTH historical training matrix through `label_cutoff`,
2. rebuild 14-day wound historical training matrix through `label_cutoff`,
3. train calibrated Tier 1 models and log them to MLflow with
   `tags.phase = production_model`,
4. build target-free scoring features for `as_of_date`,
5. score ML probabilities and point-in-time business rules,
6. compute composite expected cost and expected avoidable cost,
7. export the action queue.

## Individual Steps

Build scoring features only:

```powershell
uv run python feature_engineering\build_scoring_features.py `
  --as-of-date 2025-02-01
```

Train latest Tier 1 production models:

```powershell
uv run python modeling\train_latest_models.py `
  --label-cutoff 2025-02-01
```

Score a target-free feature file:

```powershell
uv run python modeling\score_composite.py `
  --features-path data\feature_store\scoring\features_2025-02-01.parquet `
  --output data\scored\composite_scores_2025-02-01.parquet `
  --model-phase production_model
```

## Action Queue Semantics

The default action policy is economic:

```text
economic_action = expected_avoidable_cost >= intervention_cost
```

The default intervention cost is USD 100 in `modeling/business_policy.py`.

The action queue includes:

| Column | Meaning |
|---|---|
| `composite_expected_cost` | Expected claim dollars |
| `expected_avoidable_cost` | Expected preventable claim dollars under policy assumptions |
| `top_reason` | Largest score contributor |
| `reason_codes` | Model/rule signals driving the score |
| `recommended_actions` | Suggested facility intervention |
| `*_probability` | Model or rule-derived event probability |
| `*_flag` | Active rare-event business-rule flags |

## Verification Run

The pipeline was verified with:

```powershell
uv run python modeling\train_latest_models.py `
  --label-cutoff 2025-02-01 `
  --skip-altercation

uv run python modeling\train_latest_models.py `
  --label-cutoff 2025-02-01 `
  --only-altercation

uv run python pipelines\run_daily_scoring.py `
  --as-of-date 2025-02-01 `
  --model-phase production_model
```

The scoring run produced:

| Artifact | Rows |
|---|---:|
| scoring features | 935 residents |
| composite scores | 935 residents |
| economic action queue | 125 residents |
