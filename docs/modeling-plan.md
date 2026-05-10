## 6. Hypotheses & Modeling Plan (Revised)

### Business Context Recap

The goal is to **reduce the frequency and severity of incidents** at insured facilities, which directly reduces claims below premiums collected.

| Incident Type | % Claims | Avg Cost | Count in Data | Time Horizon |
|---|---|---|---|---|
| Falls | ~13% | ~$3,500 | **2,505** ✓ | **7 days** |
| Medication errors | ~10% | ~$5,000 | 44 (low) | **7 days** |
| Wounds / pressure injuries | ~7% | ~$4,000 | **589** ✓ | **14 days** |
| Return-to-hospital (RTH) | ~7% | **~$20,000** | **1,816 transfers** ✓ | **7 days** |
| Elopement / wandering | ~5% | ~$2,500 | 7 (too low) | **30 days** |
| Altercations | ~2% | ~$2,500 | **228** ✓ | **7 days** |

---

### Class Imbalance Under New Time Horizons

With **weekly observation windows** (vitals window: Jul 2023–Jan 2025, ~82 weeks, ~3,000 residents):

| Target | Events | Obs. Windows (approx) | Positive Rate | Neg:Pos | ML Feasibility |
|---|---|---|---|---|---|
| Falls (7d) | 2,454 | ~246,000 | ~1.0% | ~100:1 | ✅ Feasible |
| RTH (7d) | 1,703 | ~246,000 | ~0.69% | ~145:1 | ✅ Feasible |
| Wounds (14d) | ~500 | ~123,000 | ~0.41% | ~245:1 | ✅ Feasible (harder) |
| Altercations (7d) | ~180 | ~246,000 | ~0.07% | ~1,400:1 | ⚠️ Borderline |
| Med Errors (7d) | ~35 | ~246,000 | ~0.014% | ~7,000:1 | ❌ Impossible |
| Choking (7d) | ~8 | ~246,000 | ~0.003% | ~30,000:1 | ❌ Impossible |
| Elopement (30d) | 7 | ~57,000 | ~0.012% | ~8,000:1 | ❌ Impossible |

**Key insight:** The shorter time horizons (7d vs. 30d) increase imbalance ~4× compared to original plan. Falls and RTH remain feasible; wounds are harder but still workable. Altercations are borderline. The remaining three types are statistically impossible for supervised ML.

---

### Modeling Tiers

#### Tier 1 — Full ML Binary Classifiers (sufficient data)

| Model | Target | Horizon | Events | Approach |
|---|---|---|---|---|
| **H1** | Fall | 7 days | 2,505 | CatBoost + SHAP |
| **H2** | RTH (unplanned transfer) | 7 days | 1,770 | CatBoost + SHAP |
| **H3** | Wound / Pressure Injury | 14 days | 589 | CatBoost + SHAP |

**Algorithm selection note:** Logistic Regression was evaluated and discarded. Its performance was materially lower than CatBoost (ROC-AUC ~0.75 vs ~0.80 for falls) and required extensive preprocessing (imputation + scaling) due to the ~45–85% null rate in vitals features. CatBoost handles missing values natively, which is the dominant data characteristic here.

These have sufficient volume and clinical feature coverage (vitals 82%, diagnoses 85%, needs 83%) for temporal binary classifiers.

---

#### Tier 2 — Hybrid: Lightweight ML + Business Rules (borderline data)

| Model | Target | Horizon | Events | Approach |
|---|---|---|---|---|
| **H4** | Altercations | 7 days | 228 | Resident-level risk score + rule-based triggers |

**Why not group with other rare events?**
- Altercations have different risk profiles (behavioral, cognitive) vs. med errors (polypharmacy, complexity) vs. choking (dysphagia) vs. elopement (dementia/wandering)
- Grouping heterogeneous targets degrades interpretability and intervention specificity
- 228 events from 127 unique residents (4.2% prevalence) is enough for a **resident-level risk classification** even if temporal window-level prediction is too sparse

**H4 Approach — Two-stage strategy:**

1. **Stage 1: Resident-level risk score (ML)** — "Is this resident prone to altercations?"
   - Observation unit: resident (not resident × window)
   - 127 positive / 3,000 total = **4.2% positive rate** → learnable
   - Features: dementia/behavioral diagnoses, `aggressive_behavior` document_tags, psychotropic medication monitoring needs, prior altercation history, ADL dependence level
   - Model: Logistic Regression or shallow GBM (regularized, few features)

2. **Stage 2: Rule-based escalation triggers** — "When should we alert for this resident?"
   - New psychotropic medication order (from `physician_orders`)
   - Medication refusal spike (from `medications`)
   - Pain level spike (from `vitals`)
   - Change in ADL/GG behavioral scores
   - Facility-level altercation cluster (temporal proximity of events at same facility)

---

#### Tier 3 — Pure Business Rules (insufficient data for ML)

For events with <50 occurrences, ML cannot learn reliable patterns. Instead, derive **rule-based risk flags** from clinical domain knowledge anchored to data columns we have:

##### Medication Errors (44 events)

| Rule | Data Source | Rationale |
|---|---|---|
| Polypharmacy flag: >15 concurrent meds | `medications` (count distinct active at time t) | More drugs → more error opportunities |
| New medication orders in last 7 days | `physician_orders` (category="Pharmacy", recent start_at) | Transition periods are highest risk |
| Cognitive impairment diagnoses | `diagnoses` (F01–F03 dementia, G30 Alzheimer's) | Patient may not self-report errors |
| "Psychotropic Medication Monitoring" need active | `needs` (need_type matching) | Already flagged by clinical staff |
| Prior medication error history | `incidents` (incident_type="Medication Error") | Recurrence pattern |
| High missed/refused rate (>15%) in last 14 days | `medications` (status aggregation) | Systemic adherence issues |

##### Choking (9 events)

| Rule | Data Source | Rationale |
|---|---|---|
| Dysphagia diagnosis active | `diagnoses` (R13.x — **465 residents** have this!) | Direct anatomical risk factor |
| Dietary texture modification order | `physician_orders` (category="Dietary - Diet") | Clinical marker for swallowing difficulty |
| Speech therapy active | `therapy_tracks` (discipline contains "ST" or "SLP") | Indicates active swallowing concern |
| Neurological diagnoses | `diagnoses` (G20 Parkinson's, I63 stroke, G30 Alzheimer's) | Neurogenic dysphagia |
| `choking` or `aspiration` document_tags | `document_tags` (tag_id matching) | Prior clinical concern documented |

##### Elopement (7 events)

| Rule | Data Source | Rationale |
|---|---|---|
| Dementia/Alzheimer's diagnosis | `diagnoses` (F01–F03, G30) | Wandering behavior associated |
| `wandering` or `elopement` document_tags | `document_tags` (tag_id matching) | Prior documented risk |
| Prior elopement history | `incidents` (incident_type="Elopement") | Strongest predictor |
| Cognitive ADL decline | `adl_responses` (Personal hygiene, cognitive activities) | Proxy for cognitive deterioration |
| New admission (first 30 days) | `residents` (admission_date) | Disorientation in new environment |

**Output format:** Each rule produces a binary flag. A resident is flagged for that specific risk when ≥N rules are active (threshold calibrated to minimize false positives while catching the known positive cases retrospectively).

---

#### Tier 4 — Composite Risk Score (Business Priority Optimization)

**Revised approach:** Combine ML model probabilities + rule-based flags into a **unified expected cost metric** per resident:

$$\text{Risk Score}_i = \hat{P}_i(\text{Fall}) \times \$3{,}500 + \hat{P}_i(\text{RTH}) \times \$20{,}000 + \hat{P}_i(\text{Wound}) \times \$4{,}000 + \text{RuleFlags}_i \times \text{AvgCost}_{\text{flag}}$$

Where:
- $\hat{P}$ comes from calibrated Tier 1 ML models (H1, H2, H3)
- Altercation risk (H4) contributes via: $\text{AltercationScore}_i \times \$2{,}500$
- Rule-based flags (Tier 3) contribute a fixed expected cost when active:
  - Med error flag active → +$5,000 × P(flag being correct)
  - Choking flag active → +$2,500 × P(flag being correct)
  - Elopement flag active → +$2,500 × P(flag being correct)

**Calibrating rule-flag probabilities:** Use retrospective precision of each rule set on historical data to estimate P(event | flag active).

**Deployment output:**
- A ranked list of residents by expected weekly claim cost
- Facility teams intervene on top quartile
- Each flagged resident gets a **reason code** (which model/rule triggered the alert)

---

### Why Not a Single Grouped "Other Incidents" Model?

| Consideration | Assessment |
|---|---|
| **Combined volume** | 228 + 44 + 9 + 7 = 288 events — still very sparse at weekly granularity |
| **Clinical coherence** | Altercations (behavioral) vs. Med errors (pharmacological) vs. Choking (anatomical) vs. Elopement (cognitive) — **completely different risk profiles** |
| **Intervention specificity** | A "something bad might happen" alert is not actionable — staff need to know WHAT to watch for |
| **Feature overlap** | Minimal — altercation features (behavioral dx, aggression tags) don't predict choking (dysphagia dx, dietary orders) |
| **Alternative: "Any Incident" model** | Dominated by falls (87% of combined events) → essentially becomes a falls model with noise |

**Conclusion:** Grouping is inferior to the tiered approach. Each event type has distinct risk factors that demand either a dedicated model (when data permits) or tailored rules (when it doesn't).

---

### Alternative Considered: Base Model + Rule Post-Processing

A potentially elegant pattern:
1. Train a **general acuity/frailty model** (predicts "any adverse event" in next 7 days)
2. Apply **incident-type-specific rules** as post-filters to differentiate alert type

**Assessment:**
- Pros: Every resident gets a base risk score; even rare events benefit from shared signal (age, comorbidity, vitals)
- Cons: The base model is essentially a falls model (74% of labels); rare event rules would need to be very high-precision to avoid alert fatigue; loses calibration per event type
- **Verdict:** Less practical than the tiered approach for deployment, but could serve as an additional "general deterioration" signal in the composite score

---

### Recommended Implementation Order

| Priority | Component | Justification |
|---|---|---|
| 1 | **H1: Falls (7d)** | Most data (2,505 events), rich feature set, dual impact (incidents + RTH reduction) |
| 2 | **H2: RTH (7d)** | Highest business value ($20k/event), strong vital signs signal |
| 3 | **Tier 3 rules** | Zero ML infrastructure needed — pure SQL/Polars logic, immediate deployability |
| 4 | **H3: Wounds (14d)** | Moderate data (589), linked to functional status and nutrition |
| 5 | **H4: Altercations** | Resident-level classification is quick to build; rule triggers add value |
| 6 | **Tier 4: Composite score** | Requires all above components; final integration + threshold tuning |

---

### Revised Data Pipeline Plan

```
1. Feature Engineering (resident × 7-day window for H1/H2, 14-day for H3)
   ├── Script: feature_engineering/build_feature_matrix.py  → feature_matrix_7d.parquet (H1, H2)
   ├── Script: feature_engineering/build_wound_matrix.py    → feature_matrix_14d.parquet (H3)
   │
   ├── Demographics: age, stay_duration, facility_id, admission_status
   ├── Diagnoses: ICD-10 chapter flags, specific fall-risk & RTH-risk codes
   ├── Vitals: rolling mean/std/min/max/count per vital_type × 3d/7d/14d lookback
   │           + binary "was_measured" flags per vital type and lookback
   ├── Care needs: open needs by category (Fall, Wound, Nutrition, Other)
   ├── History: prior incident counts by type × 7d/30d/90d/all-time, prior RTH count
   ├── Document tags: binary flags for 16 risk-relevant clinical tags
   └── Lab reports: total/abnormal/critical counts in last 14d, 30d
   │
   │   Note: Medications (16% coverage), ADL/GG responses (3-4%), therapy tracks (7%)
   │   were excluded from the initial feature set due to low population coverage.

2. Label Creation
   ├── fall_7d:             any Fall in [t, t+7d)
   ├── rth_7d:              any unplanned hospital_transfer in [t, t+7d)
   ├── wound_14d:           any Wound incident in [t, t+14d)
   └── has_altercation:     resident-level binary (for H4)

3. Temporal Split
   ├── Train/CV: observation windows where window_end <= fold_boundary (expanding window)
   ├── Holdout:  Jan 2025 (window_start >= 2025-01-01)
   └── Purging:  filter by window_end (not window_start) to exclude label windows crossing boundary

4. Modeling
   ├── H1, H2, H3: CatBoost binary classifiers (handles nulls natively — no imputation needed)
   │               Script: modeling/train_models.py
   ├── H4: CatBoost resident-level binary classifier (Stratified 5-Fold, not temporal)
   │       Script: modeling/train_altercation.py
   ├── Tier 3: Rule-based flags with retrospective validation
   │           Script: modeling/business_rules.py
   ├── Calibration: CalibratedClassifierCV(method="isotonic", cv=5) on final model
   └── Experiment tracking: MLflow (mlruns/, file backend)

5. Composite Score
   ├── Calibrate ML probabilities (isotonic regression — implemented in train_models.py)
   ├── Combine with rule-flag expected costs (formula in Section 7.6)
   └── Produce ranked resident list with reason codes
```

## 6b. Temporal Design — Feature Construction, Splitting & Retraining

### The Core Challenge

We have **19 months of full-signal data** (Jul 2023 – Jan 2025, when vitals are available). With 7-day prediction horizons, every design decision around temporal handling directly impacts:
- **Leakage risk** — using future information in features
- **Effective sample size** — how many independent observations we generate
- **Generalization** — whether the model learns patterns that hold forward in time
- **Production fidelity** — whether offline evaluation matches real-world performance

---

### 1. Point-in-Time Feature Construction

**Fundamental rule:** At observation time $t$, features may only use data with timestamps $\leq t - \text{gap}$.

#### Prediction Gap (embargo period)

A **1-day gap** between the last feature data and the start of the prediction window accounts for:
- Data entry lag (vitals/medications often entered hours after actual event)
- Intervention response time (staff need time to receive and act on alerts)
- Prevents "predicting with the answer" (e.g., a fall documented at t could have vital sign changes at t-1h that are entered retroactively)

```
Timeline for a 7-day model (e.g., Falls):

    ◄─── Feature lookback ───►│ gap │◄── Prediction window ──►
    ─────────────────────────── t-1d   t ─────────────────────── t+7d
                                 │     │                          │
                          Last   │     │  Label = 1 if any       │
                          data   │     │  Fall in [t, t+7d]      │
                          used   │     │                          │
```

#### Feature Lookback Windows

Different feature groups require different lookback depths:

| Feature Group | Lookback Windows | Rationale |
|---|---|---|
| **Vitals** (rolling stats) | 3d, 7d, 14d | Acute deterioration (3d) vs. trends (7/14d) |
| **Vitals** (trend slope) | 7d, 14d | Linear regression slope captures directionality |
| **Diagnoses** | Point-in-time snapshot | Active at time t: `onset_at <= t AND (resolved_at > t OR resolved_at IS NULL)` |
| **Incident history** | 7d, 30d, 90d, all-time | Recency matters — recent falls are strongest predictor of next fall |
| **RTH history** | 30d, 90d, all-time | Prior hospitalization pattern |
| **Medications** | 7d, 14d | Recent adherence rates (missed/refused) |
| **New orders** | 7d | Transition risk (new meds, diet changes) |
| **Lab reports** | 14d, 30d | Abnormal/critical lab counts |
| **Care needs** | Point-in-time snapshot | Open needs at time t by category |
| **ADL/GG** | Latest assessment before t | Most recent functional status |
| **Demographics** | Static | Age at t, days since admission, facility |

#### Observation Frequency: How Often Do We "Observe" Each Resident?

**Option A: Non-overlapping windows (1 obs per resident per week)**
- Pros: No autocorrelation between observations, clean independence assumption
- Cons: Fewer training samples (~246k for 7d models)
- Labels: Each window is independent — a fall counts in exactly one observation

**Option B: Daily sliding windows (1 obs per resident per day)**
- Pros: ~5× more data
- Cons: Adjacent observations are nearly identical (heavy temporal autocorrelation); labels overlap (a fall on day 4 creates positives for obs at days 1–4); inflates apparent sample size
- Risk: Model overfits to autocorrelation, not actual predictive signal

**Option C: Semi-overlapping (stride = 3–4 days for 7d models)**
- Pros: ~2× more data than Option A, moderate autocorrelation
- Cons: Still some label overlap

**Decision: Option A (non-overlapping) for training, daily scoring in production.**

Rationale:
- Clean statistical properties (no label leakage between adjacent observations)
- Avoids overestimating performance from autocorrelation
- In production, we score daily but the MODEL was trained on independent snapshots
- If data volume is insufficient, can relax to Option C (stride=4 days) as a sensitivity check

For **wound model (14d horizon):** stride = 14 days (bi-weekly non-overlapping windows).

---

### 2. Temporal Train/Test Split

#### Available Full-Signal Window

```
Jul 2023 ──────────────────────────────────────────── Jan 2025
│◄──────────────── 82 weeks total ────────────────────►│
│                                                       │
│   Vitals, Medications, GG start    ADL starts         │
│         (Jul 2023)                 (Apr 2024)         │
```

#### Split Strategy: Expanding Window Cross-Validation + Final Hold-Out

**Why not a single train/test split?**
- Only 82 weeks of data — a single split wastes either training or test data
- Temporal CV gives more robust performance estimates
- Detects if model performance degrades over time (distribution shift)

**Temporal CV scheme (expanding window, ~14 folds):**

```
Fold  1:  Train [Jul'23 ── Jan'24)   │  Test [Jan'24 w1, Jan'24 w1+7d)
Fold  2:  Train [Jul'23 ── Fev'24)   │  Test [Fev'24 w1, Fev'24 w1+7d)
...
Fold 14:  Train [Jul'23 ── Dez'24)   │  Test [Dez'24 wN, Dez'24 wN+7d)

Final:    Train [Jul'23 ──────── Dez'24]   │  Holdout [Jan'25]
```

Parameters: `MIN_TRAIN_WEEKS=26` (first boundary after 6 months of data), `FOLD_STEP=4` (one fold per 4 weeks ≈ monthly cadence). Each test window is **exactly 7 days** wide — matching the prediction horizon and the production scoring cadence.

**Why per-week test windows (not 3-month blocks)?**
This design exactly mirrors production: at any given Monday, the model is trained on all historical completed windows and predicts the next 7 days. A 3-month test block would evaluate the model on a task it was never trained for (predicting a quarter ahead). The per-fold metrics are inherently noisy (few events per week), so the primary evaluation metric is the **pooled ROC-AUC** computed on all fold predictions concatenated.

- **Final holdout (Jan 2025):** Completely unseen month, simulates first production deployment
- **`window_start` structure discovered:** Dates are staggered per-resident (aligned to each resident's admission date), not to a common weekly grid. Fold test windows therefore capture all residents whose observation window falls within the 7-day test period.

#### Why Expanding (not Sliding) Window?

- **Expanding:** Each fold adds more history to training → better use of limited data
- **Sliding:** Fixed-size window → discards oldest data → fewer samples
- With only 82 weeks, sliding would leave too little training data in early folds
- Expanding also implicitly tests whether older data is still informative (if Fold 1 performs well with only 9 months of training, the signal is strong)

#### Purging & Embargo

**Investigation findings (implemented in `modeling/train_models.py`):**

Because `window_start` is staggered per-resident (each resident's windows are anchored to their admission date), the naive filter `window_start < boundary` allows training rows whose **label window crosses the boundary**. For example, with `boundary = 2024-07-01`:

```
Resident X:
  Last train row:  window_start=Jun-29  window_end=Jul-06   ← label covers Jul 1–5
  First test row:  window_start=Jul-06  window_end=Jul-13   ← label covers Jul 6–12
```

- **Risco 1 — Mesmo incidente em train e test labels:** ❌ Não existe. As janelas são non-overlapping por residente (stride=7d), portanto nenhum evento pode pertencer simultaneamente ao label de treino e ao label de teste.
- **Risco 2 — Label de treino usa eventos futuros:** ✅ Existe. A linha `window_start=Jun-29` usa eventos de Jul 1–5 no seu label, mas esses eventos ainda não teriam ocorrido no momento do scoring (boundary = Jul 1). Empíricamente: **~840 linhas (~2.2% do treino)** por fold têm `window_end > boundary`.

**Solução implementada — purging por `window_end`:**

```python
# Remover do treino qualquer linha cuja janela de label
# ultrapasse o boundary. Garante que apenas labels completamente
# observados antes do scoring date entrem no treino.
train_df = df.filter(pl.col("window_end") <= boundary)
```

Este filtro substitui `window_start < boundary` em **todos os splits** (CV e modelo final). Custo: ~2.2% de dados de treino por fold, que são justamente as linhas mais recentes — uma troca conservadora e correcta.

---

### 3. Production Inference & Retraining Schedule

#### Inference Cadence (how often we score residents)

| Model | Prediction Horizon | Scoring Frequency | Rationale |
|---|---|---|---|
| H1: Falls | 7 days | **Daily** | Risk changes rapidly with vital signs; staff need fresh alerts |
| H2: RTH | 7 days | **Daily** | Clinical deterioration is acute; early detection is key |
| H3: Wounds | 14 days | **Twice per week** | Slower-developing; less urgency for daily scoring |
| H4: Altercations | Resident-level | **Weekly** | Risk profile changes slowly |
| Tier 3: Rules | Event-triggered | **Real-time** on data update | E.g., new Rx order triggers med error rule check |

#### Retraining Cadence

| Model | Retraining Frequency | Trigger | Rationale |
|---|---|---|---|
| H1: Falls | **Monthly** | Calendar + performance monitoring | Population turnover is ~30d (median LOS); model sees "new" residents quickly |
| H2: RTH | **Monthly** | Calendar + performance monitoring | Seasonal patterns (respiratory season) require adaptation |
| H3: Wounds | **Monthly** | Calendar | Slower events but still needs seasonal adaptation |
| H4: Altercations | **Quarterly** | Calendar | Resident-level risk evolves slowly |
| Tier 3: Rules | **No retraining** | Manual review quarterly | Deterministic rules — review thresholds based on precision |

#### Performance Monitoring (Drift Detection)

In production, monitor for model degradation:

| Signal | Metric | Alert Threshold |
|---|---|---|
| **Prediction drift** | Mean predicted probability over 7-day rolling window | >2σ shift from training-period mean |
| **Label drift** | Observed event rate vs. historical base rate | >30% relative change sustained for 2+ weeks |
| **Feature drift** | PSI (Population Stability Index) on top 10 features | PSI > 0.2 on any feature |
| **Calibration drift** | Expected vs. observed events in decile bins | Hosmer-Lemeshow p < 0.05 |
| **Performance decay** | AUPRC on rolling 30-day window | >15% relative drop from baseline |

**When drift is detected:** Trigger early retraining (don't wait for scheduled cadence).

#### Retraining Data Window

For each retrain cycle:
- **Training data:** All available historical data from Jul 2023 to current - 7 days (expanding window)
- **Validation:** Last 30 days before deployment (freshness check)
- **Why expanding, not sliding?** More data = more robust model, especially for rare events (wounds, altercations). Older patterns (mobility diagnoses → falls) remain valid even as population changes.
- **Exception:** If performance monitoring shows old data is HURTING (rare), switch to sliding window with last 12 months

#### Cold Start Problem

New residents (admitted in the last 7 days) have:
- No vitals history (or very few measurements)
- No incident history at this facility
- Diagnoses transferred from prior facility (available on admission)

**Strategy for cold-start residents:**
- Use admission diagnoses + demographics + facility-level base rate as initial risk score
- Flag as "insufficient data" for first 7 days (don't suppress alerts, but lower confidence)
- After 7 days of vitals, full model prediction kicks in
- The model handles this natively if trained with missing features (LightGBM's native missing support)

---

### 4. Summary: Complete Temporal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (offline)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Full-signal window: Jul 2023 ────────────────────── Jan 2025    │
│                                                                   │
│  Observation generation:                                          │
│    • 7d models: stride=7d → ~82 obs/resident → ~246k total       │
│    • 14d model: stride=14d → ~41 obs/resident → ~123k total      │
│                                                                   │
│  Feature construction (at each obs time t):                       │
│    • Use data with timestamp ≤ t - 1 day (embargo)               │
│    • Lookback: 3d/7d/14d for vitals, 30d/90d for history         │
│    • Point-in-time: active diagnoses, open needs                  │
│                                                                   │
│  Label construction:                                              │
│    • fall_7d: any Fall in [t, t+7d]                               │
│    • rth_7d: any unplanned transfer in [t, t+7d]                  │
│    • wound_14d: any Wound in [t, t+14d]                           │
│                                                                   │
│  Temporal CV: 3-fold expanding window + 1-week boundary embargo   │
│  Final test: Jan 2025 (held-out month)                            │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                   PRODUCTION PHASE (online)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Scoring: Daily batch (overnight) for all active residents        │
│  Output: Risk score + reason codes per resident                   │
│  Retraining: Monthly (H1, H2, H3) / Quarterly (H4)               │
│  Monitoring: Continuous drift detection (PSI, calibration, AUPRC) │
│  Cold start: Degraded-but-functional prediction using dx + demo   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Results

### 7.1 What Was Built

| Component | Script | Status |
|---|---|---|
| Feature matrix — 7d (falls, RTH) | `feature_engineering/build_feature_matrix.py` | ✅ Done |
| Feature matrix — 14d (wounds) | `feature_engineering/build_wound_matrix.py` | ✅ Done || Tier 1 ML models (H1, H2, H3) | `modeling/train_models.py` | ✅ Done |
| H4 — Altercation classifier | `modeling/train_altercation.py` | ✅ Done |
| Tier 3 — Business rules | `modeling/business_rules.py` | ✅ Done |

**Note on algorithm:** All ML models use **CatBoost**, not LightGBM as originally planned. CatBoost was selected because it handles missing values natively (critical given the 45–85% null rate in vitals features), and it consistently outperformed Logistic Regression (~ROC-AUC 0.75 vs 0.80 for falls) without requiring imputation or scaling.

---

### 7.2 Feature Matrices — Actual Output

| Matrix | File | Rows | Features | Positive Rate |
|---|---|---|---|---|
| 7-day windows | `data/feature_store/feature_matrix_7d.parquet` | 66,312 | 211 | fall_7d: 2.4%, rth_7d: 0.8% |
| 14-day windows | `data/feature_store/feature_matrix_14d.parquet` | 32,361 | 211 | wound_14d: 1.31% |

**Key implementation details:**
- **Embargo period:** 1 day between last feature data and prediction window start (prevents entry-lag leakage)
- **Purging:** Training rows filtered by `window_end <= boundary` (not `window_start`), removing ~2.2% of rows per fold whose label windows crossed fold boundaries
- **Vitals:** Filtered for `strikeout == False`; null-heavy columns (45–85% missing) passed directly to CatBoost's native missing-value handler
- **Feature lookbacks:** Vitals → 3d/7d/14d rolling stats; incident history → 7d/30d/90d/all-time; diagnoses/needs → point-in-time snapshot

---

### 7.3 Tier 1 ML Results — Expanding-Window Temporal CV

All three models use **14-fold expanding-window CV** (`MIN_TRAIN_WEEKS=26`, `FOLD_STEP=4 weeks`) plus a **January 2025 holdout set**. Calibration uses `CalibratedClassifierCV(method="isotonic", cv=5)` fitted on the full training set.

#### H1 — Falls (7-day horizon)

| Phase | ROC-AUC | Brier Score | PR-AUC |
|---|---|---|---|
| CV pooled | 0.7962 | 0.0264 | 0.1391 |
| CV per-fold | 0.7970 ± 0.0534 | — | — |
| Holdout — uncalibrated | 0.8093 | 0.0266 | — |
| Holdout — calibrated | **0.8103** | 0.0272 | — |

Calibration marginally improves discrimination on holdout. Per-fold std of ±0.053 indicates moderate temporal variability — expected with ~20–25 positive events per weekly test fold.

#### H2 — RTH (7-day horizon)

| Phase | ROC-AUC | Brier Score | PR-AUC |
|---|---|---|---|
| CV pooled | 0.7556 | 0.0092 | 0.0281 |
| CV per-fold | 0.7636 ± 0.0918 | — | — |
| Holdout — uncalibrated | 0.7371 | 0.0079 | — |
| Holdout — calibrated | **0.7342** | 0.0079 | — |

Highest per-fold variance (±0.092) due to small event counts per fold (4–14 positives). Low PR-AUC (0.028) reflects the severe class imbalance (~0.8% positive rate). Despite the Brier Score being very low, this is dominated by the large number of true negatives — the model struggles to produce high-precision alerts for RTH.

#### H3 — Wounds (14-day horizon)

| Phase | ROC-AUC | Brier Score | PR-AUC |
|---|---|---|---|
| CV pooled | 0.7805 | 0.0157 | 0.0867 |
| CV per-fold | 0.7843 ± 0.0735 | — | — |
| Holdout — uncalibrated | 0.8354 | 0.0115 | — |
| Holdout — calibrated | **0.8605** | 0.0115 | — |

Calibration provides the largest lift among the three models (+0.025 ROC-AUC on holdout). The 14-day horizon provides more signal time, reflected in the lower CV variance relative to RTH despite similar event counts.

---

### 7.4 H4 — Altercation Classifier (Resident-Level)

**Approach:** Stratified 5-Fold CV on resident-level binary classification (not temporal windows). 3,000 residents, 127 positives (4.23% positive rate). 39 features.

**Why resident-level:** The 7-day window representation would produce a ~1,400:1 imbalance (~0.07%), which is statistically impossible for reliable ML. The resident-level framing (4.23% positive) is learnable while still clinically meaningful — it identifies residents *prone* to altercations, which is actionable for care planning.

| Phase | ROC-AUC | Brier Score | PR-AUC |
|---|---|---|---|
| CV pooled | 0.8837 | 0.0342 | 0.2995 |
| CV per-fold | 0.8862 ± 0.0376 | — | — |
| OOF — uncalibrated | **0.8837** | 0.0342 | — |
| OOF — calibrated | 0.8659 | 0.0350 | — |

Uncalibrated model outperforms calibrated on this task (calibration slightly reduces discrimination with only 127 positives for the isotonic regression to fit).

**Top feature groups (by importance):** Diagnosis flags (dementia, behavioral, mood), document tags (aggressive_behavior, mental_status, psychotropic_medications), care plan needs (psychotropic, behavioral), demographics (age, los_days).

**Data leakage found and corrected:** An initial run returned ROC-AUC = 1.0000 across all folds. Investigation revealed two leaking features:
- `hist_altercation_total` — direct count of altercation incidents, which encodes the target
- `tag_altercation_incident` — document tag directly tied to altercation events

Both were removed. Additionally, `hist_incident_total` was recomputed excluding altercation events to prevent indirect leakage. After correction, ROC-AUC dropped to the realistic 0.8837.

---

### 7.5 Tier 3 — Business Rules Retrospective Validation

All rule sets were validated retrospectively on the full 3,000-resident dataset. Ground truth is derived from the `incidents` table. Results as of the full data period (Jul 2023 – Feb 2025):

#### Medication Errors (threshold: ≥ 2 of 4 rules)

| Rule | Residents Flagged | Coverage | TPs Captured |
|---|---|---|---|
| Polypharmacy (≥9 distinct meds) | ~1,800+ | >60% | — |
| Missed/refused rate >10% | — | — | — |
| Cognitive impairment dx (F01-F03, F05, G30, G31) | — | — | — |
| Prior medication error history | — | — | — |
| **Composite flag (≥2 rules)** | **240** | **8.0%** | **23 of 40 actual** |

| Metric | Value |
|---|---|
| Flagged (coverage) | 240 residents (8.0%) |
| True positives | 23 |
| Precision (PPV) | 9.58% |
| Recall | 57.50% |

**Assessment:** Captures the majority of known cases (57.5% recall) with a small footprint (8.0% of residents). Polypharmacy is the dominant trigger. Precision is low by clinical standards but typical for screening rules with rare events (~1.3% base rate).

#### Choking (threshold: ≥ 3 of 5 rules)

| Rule | Residents Flagged | Coverage | TPs Captured |
|---|---|---|---|
| Dysphagia dx (R13.x) | 465 | 15.5% | — |
| Dietary texture modification order | — | — | — |
| Speech therapy document tag | — | — | — |
| Neurological dx (G20/G30/I63/G40/G35/F03) | — | — | — |
| Choking/aspiration document tags | — | — | — |
| **Composite flag (≥3 rules)** | **734** | **24.5%** | **3 of 8 actual** |

| Metric | Value |
|---|---|
| Flagged (coverage) | 734 residents (24.5%) |
| True positives | 3 |
| Precision (PPV) | 0.41% |
| Recall | 37.50% |

**Assessment:** Threshold was raised from ≥2 to ≥3 rules after initial run at ≥2 flagged 46.7% of residents (alert fatigue risk). At ≥3 rules, coverage drops to 24.5% but recall falls to 37.5%. The 8 choking events are too few for reliable threshold calibration — the rule set is best used as a clinical screening tool rather than a predictive model. Dysphagia diagnosis (465 residents) is the dominant driver of coverage.

#### Elopement (threshold: ≥ 2 of 4 rules)

| Rule | Residents Flagged | Coverage | TPs Captured |
|---|---|---|---|
| Dementia/Alzheimer's dx (F01-F03, G30) | — | — | — |
| Wandering/elopement document tags | — | — | — |
| Prior elopement history | — | — | — |
| New admission (≤90 days of data) | — | — | — |
| **Composite flag (≥2 rules)** | **223** | **7.4%** | **5 of 6 actual** |

| Metric | Value |
|---|---|
| Flagged (coverage) | 223 residents (7.4%) |
| True positives | 5 |
| Precision (PPV) | 2.24% |
| Recall | 83.33% |

**Assessment:** Best recall among the three rule sets (83.3%) with a compact 7.4% coverage. Prior elopement history is the strongest single predictor — nearly all residents with a prior event also meet a second rule. The dementia diagnosis flag is the dominant population driver.

#### Business Rules — Summary

| Rule Set | Threshold | Flagged | Precision | Recall | Coverage |
|---|---|---|---|---|---|
| Medication Errors | ≥2 of 4 rules | 240 | 9.58% | 57.50% | 8.0% |
| Choking | ≥3 of 5 rules | 734 | 0.41% | 37.50% | 24.5% |
| Elopement | ≥2 of 4 rules | 223 | 2.24% | 83.33% | 7.4% |

**General observation:** Business rules achieve high recall (57–83%) for the events they cover, at the cost of low precision. This is the expected trade-off for rare-event screening — the goal is to not miss at-risk residents, not to avoid false alerts. Thresholds are set conservatively (low coverage) to prevent alert fatigue.

---

### 7.6 Composite Risk Score (Tier 4) — Status

The Tier 4 composite risk score (combining calibrated ML probabilities with rule-flag expected costs) was **not implemented** during this project cycle. All prerequisite components are complete:

- ✅ Calibrated probability outputs from H1, H2, H3 (MLflow-tracked)
- ✅ H4 resident-level altercation score
- ✅ Tier 3 binary rule flags

The composite scoring formula defined in Section 6 can be applied directly:

$$\text{RiskScore}_i = \hat{P}_i(\text{Fall}) \times 3{,}500 + \hat{P}_i(\text{RTH}) \times 20{,}000 + \hat{P}_i(\text{Wound}) \times 4{,}000 + \hat{P}_i(\text{Altercation}) \times 2{,}500 + \sum_k \text{RuleFlag}_{i,k} \times \text{EstCost}_k$$

Where estimated costs for rule flags use retrospective precision as P(event | flag active):
- Med error flag: $5,000 × 0.0958 ≈ $479 expected cost per flagged resident
- Choking flag: $2,500 × 0.0041 ≈ $10 expected cost per flagged resident
- Elopement flag: $2,500 × 0.0224 ≈ $56 expected cost per flagged resident

**Implementation path:** Load latest parquet matrices → run all model `predict_proba()` → join rule flags → compute composite → rank residents descending by expected cost → output reason codes per resident.

---

### 7.7 Experiment Tracking

All model runs are logged to MLflow at `mlruns/` (file-based backend). To browse:

```bash
cd <project_root>
mlflow ui --backend-store-uri mlruns
```

| Experiment | Runs Logged |
|---|---|
| `incident_prediction` | CV run + uncalibrated final + calibrated final + calibration comparison × 3 targets (H1, H2, H3) |
| `incident_prediction` | CV run + uncalibrated OOF + calibrated OOF (H4 altercation) |

Each run logs: hyperparameters, CV metrics, holdout/OOF metrics, feature importance plot, ROC/PR/calibration curves, and the serialized model artifact.