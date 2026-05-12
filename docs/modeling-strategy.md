# Modeling Strategy: Incident Risk, Action Thresholds, and Financial Backtest

## 1. Business Objective

Tricura's underwriting problem is not simply to predict incidents. The business
objective is to reduce claims severity and frequency enough that realized claims
stay below collected premiums. The modeling system therefore estimates:

1. resident-level near-term incident risk,
2. expected claim dollars associated with that risk,
3. operational action thresholds, and
4. backtested financial value under explicit intervention assumptions.

Let resident-window `i` be a resident at an observation time `t_i`. For each
incident type `j`, the model estimates:

$$
\hat{p}_{ij} = P(Y_{ij}=1 \mid X_i, \mathcal{F}_{t_i})
$$

where `X_i` contains only information available before the feature cutoff and
$\mathcal{F}_{t_i}$ is the historical information set available at scoring time.

The financial target is expected claim exposure:

$$
E[\mathrm{ClaimCost}_i]
= \sum_{j \in \mathcal{J}} \hat{p}_{ij} C_j
$$

where `C_j` is the average claim cost for incident type `j`.

## 2. Temporal Prediction Design

The production-like unit is a resident-window:

$$
i = (r, t)
$$

with resident `r` scored at observation time `t`.

The Tier 1 ML targets use these horizons:

| Target | Horizon | Label |
|---|---:|---|
| Fall | 7 days | `fall_7d` |
| Return to hospital | 7 days | `rth_7d` |
| Wound / pressure injury | 14 days | `wound_14d` |

Each label is:

$$
Y_{ij} =
\mathbb{1}\{\exists \ \mathrm{event\ of\ type}\ j
\ \mathrm{in}\ [t_i, t_i + h_j)\}
$$

A one-day embargo protects against data-entry lag and leakage:

$$
X_i = f(\mathrm{records\ with\ timestamp} \leq t_i - 1\ \mathrm{day})
$$

Training uses expanding-window temporal validation. For a fold boundary `b`,
training rows must satisfy:

$$
\mathrm{window\_end}_i \leq b
$$

and test rows satisfy:

$$
b \leq \mathrm{window\_start}_i < b + h_j
$$

This purge is important because filtering by `window_start < b` would allow
training labels whose outcome window crosses into the test period.

## 3. Why a Tiered Strategy

The data does not support the same modeling approach for all incident types.
Some events have enough examples for supervised temporal classification, while
others are too rare to estimate a reliable conditional probability from data.

The implemented strategy is:

| Tier | Incident types | Method | Rationale |
|---|---|---|---|
| Tier 1 | Falls, RTH, wounds | Temporal binary classifiers | Enough positive events and clinically relevant features |
| Tier 2 | Altercations | Resident-level classifier plus operational triggers | Weekly labels are too sparse, but resident propensity is learnable |
| Tier 3 | Medication errors, elopement | Point-in-time business rules | Listed claim categories with event counts too low for supervised ML |
| Tier 4 | All risks | Composite expected-cost score | Converts risk into business priority |

### 3.1 Why Not ML for Every Incident Type

For rare events, a supervised model would estimate:

$$
\hat{p}(Y=1 \mid X)
$$

from very few positive examples. At weekly granularity, medication errors and
elopement have expected positive rates on the order of:

$$
p \approx 10^{-4} \ \mathrm{to}\ 10^{-5}
$$

That creates several technical problems:

1. The model sees too few positive examples per temporal fold.
2. Probability calibration becomes unstable because the tail of the score
   distribution contains very few observed positives.
3. Optimizing log loss or AUC can produce a model that ranks negatives well but
   has no useful clinical action signal.
4. Feature attribution becomes misleading because tiny changes in which rare
   cases fall into a fold can dominate the learned pattern.

Business rules are therefore not a shortcut. They are the statistically
appropriate choice for extremely rare, clinically coherent risks where domain
knowledge is stronger than the observed label volume.

## 4. Tier 1: Temporal ML Classifiers

The Tier 1 models are CatBoost binary classifiers:

$$
\hat{p}_{ij} = g_j(X_i)
$$

where `j` is one of:

$$
j \in \{\mathrm{fall}, \mathrm{rth}, \mathrm{wound}\}
$$

CatBoost was selected because the feature matrices have substantial missingness
in vitals-derived features, and CatBoost can use missing values natively without
requiring imputation and scaling.

The current model matrices are:

| Matrix | Rows | Features | Targets |
|---|---:|---:|---|
| `model_matrix_7d.parquet` | 66,312 | 223 | `fall_7d`, `rth_7d` |
| `model_matrix_14d.parquet` | 32,361 | 223 | `wound_14d` |

The feature groups include demographics, vitals rolling statistics, diagnosis
flags, prior incident counts, prior RTH counts, care-plan needs, lab counts, and
document-tag indicators. They also include leakage-guarded hospital admission
context: active admission status, recent admission counts, and recent
`hospital_stay_to` counts where both the source row and event dates are known by
`feature_cutoff`.

Final holdout performance from the modeling plan:

| Model | Holdout ROC-AUC | Notes |
|---|---:|---|
| Fall 7d | 0.8127 | Strongest volume and stable signal |
| RTH 7d | 0.7529 | Improved with admission-context features but still rare |
| Wound 14d | 0.8767 | Best holdout discrimination |

The calibrated probability is then converted to expected claim exposure:

$$
E_{ij} = \hat{p}_{ij} C_j
$$

with average claim costs:

| Event type | Cost |
|---|---:|
| Fall | USD 3,500 |
| RTH | USD 20,000 |
| Wound | USD 4,000 |

## 5. Tier 2: Altercation Propensity Model

Altercations have enough resident-level positives but not enough
resident-window positives. The weekly target is approximately:

$$
P(\mathrm{altercation\ in\ next\ 7d}) \approx 0.0024
$$

so a temporal classifier would be dominated by negatives.

Instead, the system learns a resident-level propensity:

$$
\hat{q}_{r,\mathrm{alt}} = P(\mathrm{resident\ ever\ has\ altercation} \mid Z_r)
$$

where `Z_r` includes behavioral diagnoses, dementia-related diagnoses,
psychotropic or behavioral care-plan needs, document tags, demographics, and
non-altercation incident history.

The current resident-level model achieves:

| Model | CV ROC-AUC | PR-AUC |
|---|---:|---:|
| Altercation resident propensity | 0.8837 | 0.2995 |

For the composite weekly score, the resident propensity is scaled to the
observed pre-holdout weekly base rate, measured as the share of historical
resident-windows with at least one altercation:

$$
\hat{p}_{i,\mathrm{alt}}
= \min\left(1,\ \hat{q}_{r,\mathrm{alt}}
\cdot
\frac{\bar{p}_{\mathrm{weekly}}}{\bar{q}_{\mathrm{resident}}}\right)
$$

In implementation this is applied as a multiplicative scaling:

$$
\hat{p}_{i,\mathrm{alt}}
= \min\left(1,\ \hat{q}_{r,\mathrm{alt}}
\cdot s_{\mathrm{alt}}\right)
,\quad
s_{\mathrm{alt}} =
\frac{\bar{p}_{\mathrm{weekly}}}{\bar{q}_{\mathrm{resident}}}
$$

This preserves relative ranking while avoiding treating an "ever during stay"
probability as a 7-day event probability.

## 6. Tier 3: Business Rules for Rare Events

For medication errors and elopement, the system uses deterministic
point-in-time rule sets. Each rule emits binary indicators:

$$
R_{ikr} \in \{0,1\}
$$

where `k` is the rare-event type and `r` is a rule within that type.

The composite rule score is:

$$
S_{ik} = \sum_{r=1}^{m_k} R_{ikr}
$$

and the event-specific flag is:

$$
F_{ik} = \mathbb{1}\{S_{ik} \geq \tau_k\}
$$

Current thresholds:

| Rule set | Threshold | Retrospective precision | Recall | Coverage |
|---|---:|---:|---:|---:|
| Medication errors | at least 2 of 4 | 9.58% | 57.50% | 8.0% |
| Elopement | at least 2 of 4 | 2.24% | 83.33% | 7.4% |

Choking appears in the raw incident table, but it is excluded from the active
composite score because the assignment's claim breakdown does not list it as a
business claim category. With only 9 active incidents and no provided claim-cost
anchor, a dedicated choking rule would add operational complexity without
materially improving the stated financial objective.

The rule probability estimate is the retrospective positive predictive value:

$$
\hat{p}_{ik} =
P(Y_k=1 \mid F_{ik}=1)
\approx \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FP}_k}
$$

and zero when the flag is inactive:

$$
\hat{p}_{ik} =
\begin{cases}
\widehat{\mathrm{PPV}}_k, & F_{ik}=1 \\
0, & F_{ik}=0
\end{cases}
$$

The expected claim contribution is:

$$
E_{ik} = \hat{p}_{ik} C_k
$$

This produces the following expected-cost contribution per active flag:

| Rule flag | Formula | Expected cost |
|---|---|---:|
| Medication error | `0.0958 * USD 5,000` | USD 479 |
| Elopement | `0.0224 * USD 2,500` | USD 56 |

The important distinction is that these rule outputs are screening signals, not
high-precision classifiers. They are most useful when paired with low-cost,
specific interventions: medication reconciliation or elopement precautions.

## 7. Composite Expected-Cost Score

For resident-window `i`, the system combines ML probabilities and rule-derived
probabilities into a single expected claim cost:

$$
\mathrm{RiskScore}_i =
\hat{p}_{i,\mathrm{fall}} C_{\mathrm{fall}}
+ \hat{p}_{i,\mathrm{rth}} C_{\mathrm{rth}}
+ \hat{p}_{i,\mathrm{wound}} C_{\mathrm{wound}}
+ \hat{p}_{i,\mathrm{alt}} C_{\mathrm{alt}}
+ \sum_{k \in \mathcal{R}} \hat{p}_{ik} C_k
$$

where:

$$
\mathcal{R} =
\{\mathrm{med\ error}, \mathrm{elopement}\}
$$

The model also computes expected avoidable cost:

$$
A_i =
\eta \cdot \mathrm{RiskScore}_i
$$

where `eta` is the assumed intervention effectiveness for the POC scenario.

Current baseline assumption:

| Parameter | Value |
|---|---:|
| Uniform intervention effectiveness | 20% |

The backtest also reports sensitivity at 15%, 20%, and 25%. This keeps
the POC simpler and avoids false precision: the project has evidence that these
incident types are operationally preventable, but it does not have
Tricura-specific causal estimates by incident type.

This assumption lives in `modeling/business_policy.py` and should be replaced by
observed effectiveness after a prospective pilot.

## 8. Business Decision Policy

An alert is financially justified if:

$$
A_i \geq K
$$

where `K` is the operational intervention/review cost. In the backtest:

$$
K = \mathrm{USD}\ 100
$$

The system evaluates two policy families:

1. capacity policies: alert on top `q%` of resident-windows within each facility,
2. economic threshold: alert when expected avoidable cost exceeds intervention
   cost.

For capacity policy `q`, the alert set is:

$$
\mathcal{A}_q =
\{i: \mathrm{rank}_{f(i)}(\mathrm{RiskScore}_i)
\leq \lceil q N_{f(i)} \rceil\}
$$

where `f(i)` is the facility for row `i` and `N_f` is the number of scored rows
for that facility.

For the economic policy:

$$
\mathcal{A}_{econ} = \{i: A_i \geq K\}
$$

The resident output table includes:

| Field | Meaning |
|---|---|
| `composite_expected_cost` | Expected claim dollars |
| `expected_avoidable_cost` | Expected claim dollars that could be avoided |
| `economic_action` | Whether `expected_avoidable_cost >= K` |
| `top_reason` | Largest expected-cost contributor |
| `reason_codes` | Model and rule signals driving the score |

## 9. Financial Backtest Methodology

The January 2025 holdout simulates a first production month:

| Quantity | Value |
|---|---:|
| Scoring period | 2025-01-01 to 2025-02-01 |
| Scored resident-windows | 3,612 |
| Actual events | 253 |
| Actual modeled claim cost | USD 1,788,500 |

An event is captured if a resident had an alert before the event and the event
fell inside that event type's horizon:

$$
\mathrm{Captured}(e, i) =
\mathbb{1}\{i \in \mathcal{A}\}
\mathbb{1}\{r_e = r_i\}
\mathbb{1}\{t_e \in [t_i, t_i + h_e)\}
$$

Events are deduplicated so the same event cannot be counted multiple times.

Captured claim exposure is:

$$
\mathrm{CapturedCost} =
\sum_{e \in \mathcal{E}}
C_e \cdot
\mathbb{1}\{\exists i: \mathrm{Captured}(e,i)=1\}
$$

Estimated avoided claim dollars are:

$$
\mathrm{AvoidedCost} =
\sum_{e \in \mathcal{E}}
C_e \eta \cdot
\mathbb{1}\{\exists i: \mathrm{Captured}(e,i)=1\}
$$

where `eta` is the uniform effectiveness scenario.

Intervention cost is:

$$
\mathrm{InterventionCost} = |\mathcal{A}| K
$$

Net savings are:

$$
\mathrm{NetSavings} =
\mathrm{AvoidedCost} - \mathrm{InterventionCost}
$$

and ROI is:

$$
\mathrm{ROI} =
\frac{\mathrm{NetSavings}}{\mathrm{InterventionCost}}
$$

## 10. Backtest Results

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
| 15% | 401 | USD 304,500 | USD 45,675 | USD 40,100 | USD 5,575 | 0.14x |
| 20% | 401 | USD 304,500 | USD 60,900 | USD 40,100 | USD 20,800 | 0.52x |
| 25% | 401 | USD 304,500 | USD 76,125 | USD 40,100 | USD 36,025 | 0.90x |

Primary policy detail (`top_10pct_per_facility`):

| Incident type | Actual events | Captured events | Captured claim cost | Capture rate |
|---|---:|---:|---:|---:|
| Return to hospital | 54 | 7 | USD 140,000 | 13.0% |
| Fall | 159 | 39 | USD 136,500 | 24.5% |
| Wound / pressure injury | 33 | 7 | USD 28,000 | 21.2% |
| Altercation | 6 | 0 | USD 0 | 0.0% |
| Medication error | 1 | 0 | USD 0 | 0.0% |

The strongest observed metric is captured claim exposure. Net savings and ROI
depend on the uniform effectiveness scenario and alert review cost.

## 11. Interpretation

The system is designed to make risk operational:

1. ML classifiers rank high-volume/high-cost incidents with calibrated
   probabilities.
2. Rare-event business rules preserve clinical actionability where ML would be
   statistically underpowered.
3. The composite score converts heterogeneous risks into expected dollars.
4. Action policies convert expected dollars into facility workloads.
5. The backtest converts alerts into measurable financial indicators.

The strongest observed metric is captured claim exposure. Estimated savings are
not causal proof; they depend on the uniform intervention-effectiveness
scenario. A
prospective pilot should randomize or stagger deployment across facilities and
measure:

$$
\frac{\mathrm{ActualClaims}}{\mathrm{ExpectedClaims}}
$$

before and after deployment, controlling for resident mix and facility baseline
risk.

## 12. Implementation Artifacts

| Artifact | Purpose |
|---|---|
| `modeling/train_models.py` | Tier 1 temporal classifiers |
| `modeling/train_altercation.py` | Tier 2 resident-level altercation model |
| `modeling/business_rules.py` | Tier 3 retrospective and point-in-time rules |
| `modeling/score_composite.py` | Composite expected-cost scoring |
| `modeling/backtest_financials.py` | Financial backtest |
| `modeling/train_latest_models.py` | Latest-data production model training |
| `modeling/business_policy.py` | Claim costs and intervention assumptions |
| `modeling/outcome_costs.py` | Actual event cost construction |
| `feature_engineering/build_scoring_features.py` | Target-free new-batch feature generation |
| `pipelines/run_daily_scoring.py` | End-to-end daily scoring workflow |
| `data/scored/composite_scores_holdout.parquet` | Scored holdout resident-windows |
| `modeling/artifacts/financial_backtest_summary.csv` | Policy-level financial results |
| `modeling/artifacts/financial_backtest_by_incident_type.csv` | Event-type financial results |
| `modeling/artifacts/financial_effectiveness_sensitivity.csv` | Uniform-effectiveness sensitivity results |
| `docs/backtest-results.md` | Generated backtest summary |
| `docs/prediction-pipeline.md` | New-batch scoring operations guide |
