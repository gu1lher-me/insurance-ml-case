# EDA Findings Reference — Tricura ML Case

> Generated from `exploration/01_eda.ipynb`. Use this document as the ground truth for data characteristics when building feature engineering and modeling notebooks.

---

## 1. Data Inventory

| Table | Rows | PK | Unique Residents | Unique Facilities | Coverage | Date Range |
|---|---|---|---|---|---|---|
| `residents` | 3,000 | `resident_id` | 3,000 | 100 | 100% | 2006–2025 |
| `incidents` | 3,578 | `incident_id` | 987 | 92 | 32.9% | 2019–2025 |
| `diagnoses` | 60,620 | `diagnosis_id` | 2,547 | 99 | 84.9% | 2019–2025 |
| `vitals` | 2,517,056 | `vital_id` | 2,464 | 89 | 82.1% | **2023–2025** |
| `hospital_transfers` | 1,816 | `transfer_id` | 911 | 88 | 30.4% | 2006–2025 |
| `hospital_admissions` | 2,945 | `admission_id` | 1,873 | 89 | 62.4% | 2006–2025 |
| `medications` | 1,430,877 | `medication_id` | 481 | 35 | 16.0% | 2023–2025 |
| `adl_responses` | 480,554 | `adl_response_id` | 96 | 9 | 3.2% | **2024–2025** |
| `gg_responses` | 660,711 | `gg_response_id` | 134 | 18 | 4.5% | 2023–2025 |
| `needs` | 162,762 | `need_id` | 2,490 | 100 | 83.0% | — |
| `factors` | 190,284 | `factor_id` | — (via `incident_id`) | 92 | — | — |
| `lab_reports` | 13,334 | `lab_report_id` | 1,171 | 60 | 39.0% | — |
| `physician_orders` | 94,051 | `order_id` | 2,399 | 90 | 80.0% | — |
| `care_plans` | 3,034 | `care_plan_id` | 2,745 | 100 | 91.5% | — |
| `injuries` | 1,219 | `injury_id` | — (via `incident_id`) | 88 | — | — |
| `therapy_tracks` | 761 | `therapy_id` | 219 | 15 | 7.3% | — |
| `document_tags` | 562,905 | `document_tag_id` | 1,018 | 36 | 33.9% | — |

**Key constraint:** The richest feature source (vitals) only starts July 2023. This limits the full-signal training window to ~19 months (Jul 2023–Jan 2025).

---

## 1b. Table Schemas & Join Keys

### Join Key Map

All tables share `resident_id` and `facility_id` as foreign keys to the `residents` master table. Two additional join keys link child tables to specific entities:

| Join Key | Parent Table | Child Tables |
|---|---|---|
| `resident_id` | `residents` | All 16 other tables |
| `facility_id` | `residents` | All 16 other tables |
| `incident_id` | `incidents` | `factors`, `injuries` |
| `care_plan_id` | `care_plans` | `needs` |

### Entity-Relationship Diagram (text)

```
residents (master)
 ├─── incidents ──┬── factors      (incident_id)
 │                └── injuries     (incident_id)
 ├─── diagnoses
 ├─── vitals
 ├─── hospital_transfers
 ├─── hospital_admissions
 ├─── medications
 ├─── adl_responses
 ├─── gg_responses
 ├─── care_plans ──── needs        (care_plan_id)
 ├─── physician_orders
 ├─── lab_reports
 ├─── therapy_tracks
 └─── document_tags
```

### Column-Level Schemas

#### `residents` — Master table (3,000 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `resident_id` | String | **PK** — 3,000 unique |
| `facility_id` | String | FK — 100 unique facilities |
| `date_of_birth` | Datetime[μs] | |
| `admission_date` | Datetime[μs] | SNF admission date |
| `discharge_date` | Datetime[μs] | Null if still admitted |
| `deceased_date` | Datetime[μs] | Null if alive |
| `outpatient` | Boolean | Small subset flagged |
| `created_at` | Datetime[μs] | |
| `updated_at` | Datetime[μs] | |

#### `incidents` — Target source (3,578 rows × 8 cols)

| Column | Type | Notes |
|---|---|---|
| `incident_id` | String | **PK** — referenced by `factors`, `injuries` |
| `facility_id` | String | FK → residents |
| `resident_id` | String | FK → residents (987 unique) |
| `incident_type` | String | 6 values: Fall, Wound, Altercation, Choking, Elopement, Medication Error |
| `incident_location` | String | 57 unique locations (1 null) |
| `occurred_at` | Datetime[μs] | Event timestamp |
| `strikeout` | Boolean | 5.5% strikeout rate |
| `created_at` | Datetime[μs] | |

#### `diagnoses` (60,620 rows × 8 cols)

| Column | Type | Notes |
|---|---|---|
| `diagnosis_id` | String | **PK** |
| `facility_id` | String | FK |
| `resident_id` | String | FK → residents (2,547 unique) |
| `icd_10_code` | String | 4,897 unique codes; 259 nulls |
| `onset_at` | Datetime[μs] | Diagnosis start date |
| `resolved_at` | Datetime[μs] | Null if still active |
| `strikeout` | Boolean | 4.2% strikeout rate |
| `created_at` | Datetime[μs] | |

#### `vitals` (2,517,056 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `vital_id` | String | **PK** |
| `resident_id` | String | FK → residents (2,464 unique) |
| `facility_id` | String | FK |
| `vital_type` | String | 8 types: BP - Systolic, Pulse, O2 sats, Blood Sugar, Temperature, Respiration, Pain Level, Weight |
| `value` | Float64 | Primary measurement — **outliers present, clipping required** |
| `dystolic_value` | Float64 | Diastolic BP (note: column name has a **typo** — "dystolic" instead of "diastolic"); populated only for BP rows |
| `measured_at` | Datetime[μs] | Measurement timestamp |
| `strikeout` | Boolean | 0.17% strikeout rate |
| `created_at` | Datetime[μs] | |

#### `hospital_transfers` — RTH source (1,816 rows × 12 cols)

| Column | Type | Notes |
|---|---|---|
| `transfer_id` | String | **PK** |
| `facility_id` | String | FK |
| `resident_id` | String | FK → residents (911 unique) |
| `effective_date` | Datetime[μs] | Transfer date |
| `ineffective_date` | Datetime[μs] | End date (return to SNF) |
| `to_from_type` | String | 15 unique values — **hospital type**, not direction. Needs case normalization (e.g. "Acute Care Hospital" vs "Acute care HOSPITAL") |
| `transfer_outcome` | String | Admitted Inpatient, ED Visit Only, etc. |
| `stay_purpose` | String | |
| `transfer_reason` | String | Altered Mental Status, Fall, Shortness of Breath, etc. |
| `planned_flag` | Boolean | 34 nulls; False/null = unplanned (97.5%) |
| `emergency_flag` | Boolean | 34 nulls; 303 emergency transfers |
| `created_at` | Datetime[μs] | |

#### `hospital_admissions` (2,945 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `admission_id` | String | **PK** |
| `facility_id` | String | FK |
| `resident_id` | String | FK → residents (1,873 unique) |
| `effective_date` | Datetime[μs] | Admission date to SNF |
| `ineffective_date` | Datetime[μs] | |
| `admission_status` | String | 3 values: "Chronic Long-Term", "Post Acute"; 79 nulls |
| `emergency_flag` | String | **100% null** — not usable |
| `hospital_stay_to` | Datetime[μs] | End of prior hospital stay |
| `created_at` | Datetime[μs] | |

#### `medications` (1,430,877 rows × 8 cols)

| Column | Type | Notes |
|---|---|---|
| `medication_id` | String | **PK** |
| `resident_id` | String | FK → residents (481 unique — 16% coverage) |
| `facility_id` | String | FK (35 facilities only) |
| `description` | String | Free-text drug name + dosage; 19,189 unique values — very high cardinality |
| `scheduled_at` | Datetime[μs] | Scheduled administration time |
| `administered_at` | Datetime[μs] | Actual administration time |
| `status` | String | On Time (56.3%), Late (35.9%), Missed (7.2%), Refused (0.7%) |
| `created_at` | Datetime[μs] | |

#### `adl_responses` (480,554 rows × 12 cols)

| Column | Type | Notes |
|---|---|---|
| `adl_response_id` | String | **PK** |
| `resident_id` | String | FK → residents (96 unique — very sparse) |
| `facility_id` | String | FK (9 facilities only) |
| `assessment_date` | Datetime[μs] | 2024-04 to 2025-01 |
| `activity` | String | 22 activities (Bed mobility, Transfer, Walking, etc.) |
| `category` | String | "Self-Performance" or "Support" |
| `response` | String | Scale 0–4, "n/a", "7" (refused) |
| `response_description` | String | Text label for the response code |
| `response_status` | String | "Complete" only |
| `previous_response` | String | Prior assessment response |
| `adl_change` | Int64 | Change from previous assessment |
| `created_at` | Datetime[μs] | |

#### `gg_responses` (660,711 rows × 10 cols)

| Column | Type | Notes |
|---|---|---|
| `gg_response_id` | String | **PK** |
| `facility_id` | String | FK (18 facilities) |
| `resident_id` | String | FK → residents (134 unique — sparse) |
| `task_group` | String | "Self Care", "Mobility" |
| `task_name` | String | 27 tasks: Sit to Stand, Eating, Oral Hygiene, etc. |
| `response` | String | Text response label |
| `response_code` | Int64 | 1=Dependent to 6=Independent, 7=Refused, 9/10=Not attempted, 88=N/A; 23,500 nulls |
| `previous_response_code` | Int64 | Prior code; 24,624 nulls |
| `change` | Int64 | Change from previous assessment |
| `created_at` | Datetime[μs] | |

#### `care_plans` (3,034 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `care_plan_id` | String | **PK** — referenced by `needs` |
| `resident_id` | String | FK → residents (2,745 unique — 91.5% coverage) |
| `facility_id` | String | FK (100 facilities) |
| `initiated_at` | Datetime[μs] | Plan start date |
| `closed_reason` | String | |
| `closed_at` | Datetime[μs] | Null if still open |
| `next_review_at` | Datetime[μs] | |
| `strikeout` | Boolean | |
| `created_at` | Datetime[μs] | |

#### `needs` (162,762 rows × 11 cols)

| Column | Type | Notes |
|---|---|---|
| `need_id` | String | **PK** |
| `resident_id` | String | FK → residents (2,490 unique — 83%) |
| `facility_id` | String | FK (100 facilities) |
| `care_plan_id` | String | FK → `care_plans` (2,757 unique) |
| `need_type` | String | 1,184 unique values; 12,846 nulls. E.g. "Falls (CAA 11)", "Nutritional Status (CAA 12)" |
| `need_category` | String | 4 values: Other, Wound, Fall, Nutrition |
| `initiated_at` | Datetime[μs] | |
| `resolved_at` | Datetime[μs] | Null if still open |
| `strikeout` | Boolean | |
| `current_row` | Boolean | |
| `created_at` | Datetime[μs] | |

#### `factors` (190,284 rows × 5 cols)

| Column | Type | Notes |
|---|---|---|
| `factor_id` | String | **PK** |
| `incident_id` | String | FK → `incidents` (3,290 unique) |
| `facility_id` | String | FK (92 facilities) |
| `factor_type` | String | 4 values: Predisposing Environmental, Predisposing Physiological, Predisposing Situation, Factors Note |
| `created_at` | Datetime[μs] | |

> **No `resident_id`** — must join through `incidents` to reach residents.

#### `injuries` (1,219 rows × 7 cols)

| Column | Type | Notes |
|---|---|---|
| `injury_id` | String | **PK** |
| `incident_id` | String | FK → `incidents` (871 unique) |
| `facility_id` | String | FK (88 facilities) |
| `injury_type` | String | 52 types: Bruise, Skin Tear, Fracture, Abrasion, etc. |
| `injury_location` | String | Body location |
| `is_post_incident` | Boolean | |
| `created_at` | Datetime[μs] | |

> **No `resident_id`** — must join through `incidents` to reach residents.

#### `lab_reports` (13,334 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `lab_report_id` | String | **PK** |
| `resident_id` | String | FK → residents (1,171 unique — 39%) |
| `facility_id` | String | FK (60 facilities) |
| `lab_name` | String | 4,121 unique lab test names — high cardinality |
| `status` | String | Completed, Resulted, Cancelled |
| `severity_status` | String | Normal, Abnormal, Critical |
| `reported_at` | Datetime[μs] | |
| `collected_at` | Datetime[μs] | |
| `created_at` | Datetime[μs] | |

#### `physician_orders` (94,051 rows × 10 cols)

| Column | Type | Notes |
|---|---|---|
| `order_id` | String | **PK** |
| `resident_id` | String | FK → residents (2,399 unique — 80%) |
| `facility_id` | String | FK (90 facilities) |
| `category` | String | 7 values: Pharmacy, Laboratory, Dietary - Diet, Dietary - Supplements, Diagnostic, Enteral - Feed, Other |
| `ordered_at` | Datetime[μs] | |
| `start_at` | Datetime[μs] | |
| `end_at` | Datetime[μs] | |
| `order_status` | String | Active, Completed, Discontinued |
| `frequency` | String | Dosing frequency |
| `created_at` | Datetime[μs] | |

#### `therapy_tracks` (761 rows × 7 cols)

| Column | Type | Notes |
|---|---|---|
| `therapy_id` | String | **PK** |
| `facility_id` | String | FK (15 facilities) |
| `resident_id` | String | FK → residents (219 unique — 7.3%) |
| `discipline` | String | OT, PT, etc. |
| `start_at` | Datetime[μs] | |
| `end_at` | Datetime[μs] | |
| `created_at` | Datetime[μs] | |

#### `document_tags` (562,905 rows × 9 cols)

| Column | Type | Notes |
|---|---|---|
| `document_tag_id` | String | **PK** |
| `resident_id` | String | FK → residents (1,018 unique — 33.9%) |
| `facility_id` | String | FK (36 facilities) |
| `doc_type` | String | 14 types: incidents, needs, assessments, factors, goals, interventions, notes, notifications, etc. |
| `tag_id` | String | 186 semantic tags: actual_fall, actual_wound, aggressive_behavior, alzheimers_disease, etc. |
| `match_confidence` | Float64 | 127,822 nulls |
| `editable` | Boolean | |
| `created_at` | Datetime[μs] | |
| `deleted_at` | Datetime[μs] | |

---

## 1c. Useful Join Patterns for Feature Engineering

### Resident-centric views (join on `resident_id`)

All feature engineering joins start from the `residents` master table. Most tables can be joined directly via `resident_id`. Key patterns:

| View | Join Path | Purpose |
|---|---|---|
| **Resident + Diagnoses** | `residents ← diagnoses` on `resident_id` | Comorbidity flags, Charlson index |
| **Resident + Vitals** | `residents ← vitals` on `resident_id` | Rolling vital statistics per window |
| **Resident + Incidents (history)** | `residents ← incidents` on `resident_id` | Prior incident counts by type |
| **Resident + RTH history** | `residents ← hospital_transfers` on `resident_id` | Prior transfer counts, reasons |
| **Resident + Admission context** | `residents ← hospital_admissions` on `resident_id` | Admission status (Chronic vs Post Acute), readmission patterns |
| **Resident + Medications** | `residents ← medications` on `resident_id` | Polypharmacy count, missed/refused rates |
| **Resident + Needs** | `residents ← needs` on `resident_id` | Open need counts by category (Fall, Wound, Nutrition) |
| **Resident + ADL** | `residents ← adl_responses` on `resident_id` | Functional status scores (sparse) |
| **Resident + GG** | `residents ← gg_responses` on `resident_id` | CMS functional assessment (sparse) |
| **Resident + Lab reports** | `residents ← lab_reports` on `resident_id` | Abnormal/critical lab counts |
| **Resident + Physician orders** | `residents ← physician_orders` on `resident_id` | Active order counts by category |
| **Resident + Therapy** | `residents ← therapy_tracks` on `resident_id` | Active therapy flag (OT/PT) |

### Incident-enrichment views (join on `incident_id`)

These extend incident records with contextual detail:

| View | Join Path | Purpose |
|---|---|---|
| **Incident + Factors** | `incidents ← factors` on `incident_id` | Predisposing factors (physiological, environmental, situational) per incident |
| **Incident + Injuries** | `incidents ← injuries` on `incident_id` | Injury severity/type for outcome analysis |
| **Incident + Factors + Residents** | `incidents ← factors` on `incident_id`, then `incidents ← residents` on `resident_id` | Resident-level factor profiles |

### Care plan views (join on `care_plan_id`)

| View | Join Path | Purpose |
|---|---|---|
| **Care plan + Needs** | `care_plans ← needs` on `care_plan_id` | Need types and resolution status per care plan |
| **Resident + Care plan + Needs** | `residents ← care_plans` on `resident_id` → `care_plans ← needs` on `care_plan_id` | Full care plan picture per resident |

### Cross-table composite views

| View | Tables | Purpose |
|---|---|---|
| **Full resident feature matrix** | `residents` + `diagnoses` + `vitals` + `incidents` + `hospital_transfers` + `needs` + `medications` (where available) | The final modeling input — one row per resident × 30-day window |
| **Document tags as NLP features** | `document_tags` filtered by `tag_id` (e.g. `actual_fall`, `actual_wound`) joined to `residents` | Augment labels or add semantic flags from clinical notes |
| **Transfer-enriched incidents** | `hospital_transfers` joined to `incidents` on `resident_id` + time proximity | Identify transfers caused by incidents (e.g. fall → ER visit) |

### Notes on sparse tables

- **`medications`** (16% coverage, 35 facilities): Only use as features for the subset of residents that have records. Set features to null/missing for others — LightGBM handles missing values natively.
- **`adl_responses`** (3.2%, 9 facilities) and **`gg_responses`** (4.5%, 18 facilities): Too sparse for population-level features. Use as supplementary features when available.
- **`factors`** and **`injuries`**: No direct `resident_id` — must join via `incidents.incident_id` first.
- **`hospital_admissions.emergency_flag`**: Column exists but is 100% null — not usable. Use `hospital_transfers.emergency_flag` instead.

---

## 2. Incidents

### Distribution by Type (active, strikeout excluded)

| Incident Type | Count | % of Active |
|---|---|---|
| Fall | 2,505 | 74.1% |
| Wound | 589 | 17.4% |
| Altercation | 228 | 6.7% |
| Medication Error | 44 | 1.3% |
| Choking | 9 | 0.3% |
| Elopement | 7 | 0.2% |

- Total records in table: **3,578**
- Active incidents (strikeout=False): **3,382**
- Strikeout rate: **5.5%**
- Date range: 2019-01-10 → 2025-01-31
- Only `incident_location` has nulls (1 null).

### Resident-Level Prevalence

| Event | # Residents | % of 3,000 |
|---|---|---|
| Any incident | 978 | 32.6% |
| Fall | 806 | 26.9% |
| Wound | 299 | 10.0% |
| Altercation | 127 | 4.2% |
| RTH transfer | 911 | 30.4% |

- Among residents with any incident: mean 3.5 incidents, median 2, max 46.

---

## 3. RTH Events (hospital_transfers)

### Key Finding: `to_from_type` is the hospital TYPE, not direction

All 1,816 records in `hospital_transfers` represent transfers **from the SNF to a hospital**. The `to_from_type` column encodes the destination hospital category (e.g., `"Acute care hospital"`, `"EMERGENCY ROOM"`), **not** whether the transfer was incoming or outgoing. There is no `"To Hospital"` value in this column.

### RTH Definition

```python
rth = hospital_transfers.filter(
    (pl.col("planned_flag") == False) | (pl.col("planned_flag").is_null())
)
```

- Unplanned transfers: **1,770 / 1,816 = 97.5%**
- Emergency transfers (`emergency_flag=True`): **303**

### Transfer Outcomes

| Outcome | Count |
|---|---|
| Admitted, Inpatient | 738 |
| null | 515 |
| ED Visit Only | 251 |
| Admitted, Status Uncertain | 149 |
| Other | 101 |
| Admitted, Observation | 62 |

### Top Transfer Reasons (excluding null/Other)

| Reason | Count |
|---|---|
| Altered Mental Status | 128 |
| Fall | 127 |
| Shortness of Breath | 87 |
| Chest Pain | ~55 |
| Fever | 27 |
| GI Bleeding | 32 |
| Unresponsive | 32 |

> Note: Fall is the #2 transfer reason — fall prevention has a **dual impact** on both incident and RTH claims.

---

## 4. Residents

- **Total:** 3,000 residents across **100 unique facilities** (avg 30 residents/facility)
- **Age at admission:** Mean ~74.4 years; right-skewed distribution typical of SNF population
- **Discharge status:** Subset are discharged or deceased (exact counts available in notebook)
- **Outpatient flag:** Small subset

### Length of Stay

- Median ~31 days; long-tail distribution (post-acute vs. chronic long-term residents)
- Two populations visible: short-stay post-acute (~30–90 days) and long-stay chronic (>180 days)

---

## 5. Diagnoses

- **Active diagnoses:** 58,067 (after filtering strikeout=False; 4.2% strikeout rate)
- **Residents with diagnoses:** 2,547 / 3,000 (84.9%)
- **Mean active diagnoses per resident:** 22.8 — high comorbidity burden typical of SNF

### Top ICD-10 Codes

| ICD-10 | Description | Count |
|---|---|---|
| M62.81 | Sarcopenia | 2,169 |
| I10 | Hypertension | 1,735 |
| E78.5 | Hyperlipidemia | 1,225 |
| Z74.1 | Personal care assistance needed | 954 |
| K21.9 | GERD | 894 |
| R26.2 | Difficulty walking | ~860 |
| R26.89 | Other gait/mobility abnormalities | 538 |
| R27.8 | Other lack of coordination | 586 |
| E11.9 | Type 2 Diabetes | 612 |
| U07.1 | COVID-19 | 605 |
| J44.9 | COPD unspecified | 588 |
| D64.9 | Anemia, unspecified | 489 |
| E03.9 | Hypothyroidism | 482 |
| R13.12 | Dysphagia | 465 |
| F32.A | Depression | 463 |
| I25.10 | Coronary artery disease | 460 |

> **Feature engineering note:** Group codes by ICD-10 chapter and use the Charlson Comorbidity Index (CCI). Raw codes are too sparse (thousands of unique values) to use as categoricals.

---

## 6. Vitals

- **Total records:** 2,517,056 (Jul 2023 – Jan 2025 only)
- **Residents covered:** 2,464 / 3,000 (82.1%)
- **Strikeout rate:** 0.17% (very low — most records are valid)

### Vital Types and Record Counts

| Vital Type | Count |
|---|---|
| Pain Level | 831,372 |
| BP - Systolic | 370,662 |
| Pulse | 336,209 |
| O2 sats | 266,805 |
| Blood Sugar | 250,563 |
| Temperature | 238,352 |
| Respiration | 189,396 |
| Weight | 33,697 |

### ⚠️ Outliers — Clipping Required

| Vital | Clinical Valid Range | Observed Max | Action |
|---|---|---|---|
| BP - Systolic | 60–250 mmHg | **13,385** | Clip to [60, 250] |
| O2 sats | 50–100% | **9,994** | Clip to [50, 100] |
| Blood Sugar | 20–600 mg/dL | Inspect | Clip to [20, 600] |
| Temperature | 90–108°F | Inspect | Clip to [90, 108] |
| Pulse | 20–200 bpm | Inspect | Clip to [20, 200] |

### BP Systolic Stats (post-strikeout filter, pre-clip)

- Mean: 128.2 mmHg | Std: 33.9 | Min: 0 | Max: 13,385
- P25: 118 | P50: 128 | P75: 136

### O2 Saturation Stats (post-strikeout filter, pre-clip)

- Mean: 96.4% | Std: 34.5 | Min: 0 | Max: 9,994
- P25: 95 | P50: 97 | P75: 97.5

---

## 7. Medications

- **Total records:** 1,430,877
- **Residents covered:** 481 / 3,000 (**16% only** — data from a subset of facilities)
- **Unique drug descriptions:** 19,189 (free-text field; very high cardinality)

### Medication Status Breakdown

| Status | Count | % |
|---|---|---|
| On Time | 805,482 | 56.3% |
| Late | 513,171 | 35.9% |
| Missed | 102,346 | 7.2% |
| Refused | 9,878 | 0.7% |

### Polypharmacy (per resident with medication records)

- Mean: 46.8 unique medications | Median: 37 | Max: 347
- Residents with >10 unique meds: 473/481 (98%)
- Residents with >20 unique meds: 409/481 (85%)

> **Note:** The "unique medications" count is inflated by free-text descriptions. Normalize by extracting drug class/generic name for true polypharmacy count. As a fallback feature, use `missed_rate` and `refused_rate` per resident per window.

---

## 8. ADL Responses (MDS Standard)

- **Total records:** 480,554
- **Residents covered:** 96 / 3,000 (**3.2% — very sparse**)
- **Date range:** 2024-04-26 → 2025-01-31 (recent only)

### Activities Available (22 total)

Bed mobility, Transfer, Walking in room/corridor, Locomotion on/off unit, Personal hygiene, Dressing, Bathing, Eating, Toilet use — each split into Self-Performance and Support scores.

### Response Scale

`0` = Independent, `1` = Supervision, `2` = Limited assistance, `3` = Extensive assistance, `4` = Total dependence, `n/a` = Activity did not occur, `7` = Activity refused

| Response | Count |
|---|---|
| n/a | 126,497 |
| 0 (Independent) | 117,846 |
| 2 | 103,916 |
| 4 (Total depend.) | 53,275 |
| 1 | 39,796 |
| 3 | 36,187 |
| 7 (Refused) | 3,037 |

### Mean Mobility ADL Score (among 85 residents with mobility data)

- Mean: 2.0 (moderate dependence) | Std: 1.3 | Min: 0.03 | Max: 3.98

---

## 9. GG Responses (CMS Functional Assessment)

- **Total records:** 660,711
- **Residents covered:** 134 / 3,000 (**4.5% — sparse**)
- **Task groups:** Self Care, Mobility

### Response Code Scale

`1` = Dependent, `2` = Maximal assistance, `3` = Moderate assistance, `4` = Minimal assistance, `5` = Supervision, `6` = Independent, `7` = Patient refused, `9` = Not attempted, `88` = Not applicable, `10` = Not attempted due to medical condition

---

## 10. Needs (Care Plan Needs)

- **Total records:** 162,762

### Need Categories

| Category | Count |
|---|---|
| Other | 127,731 |
| Wound | 13,984 |
| Fall | 11,651 |
| Nutrition | 9,396 |

> The `Fall` and `Wound` need categories map directly to clinical care plan assessments. Count of open `Fall`-category needs at observation time is a strong label-correlated feature.

### Top Need Types (selected)

- Medications - Psychotropic Medication Monitoring
- Nutritional Status (CAA 12)
- Falls (CAA 11)
- ADL/Mobility — Functional / Rehabilitation
- Immunological (Infections)
- Respiratory/Pulmonary
- Pain (CAA 19)

---

## 11. Class Imbalance Estimates (Revised — per time horizon)

Computed for the **Jul 2023–Jan 2025 window** with the **revised time horizons** (7d/14d/30d). Observation windows are non-overlapping.

### Weekly observation windows (~246,000 resident-weeks)

| Model Target | Horizon | Events in Window | Positive Rate | Neg:Pos | ML Feasibility |
|---|---|---|---|---|---|
| Fall (H1) | 7 days | 2,454 | ~1.0% | ~100:1 | ✅ Feasible |
| RTH (H2) | 7 days | 1,703 | ~0.69% | ~145:1 | ✅ Feasible |
| Altercations (H4) | 7 days | 148 positive windows | ~0.24% | ~417:1 | ⚠️ Borderline |
| Med Errors | 7 days | ~35 | ~0.014% | ~7,000:1 | ❌ Impossible |
| Choking | 7 days | ~8 | ~0.003% | ~30,000:1 | Excluded from claim-backed model |

### Bi-weekly observation windows (~123,000 resident-fortnights)

| Model Target | Horizon | Events in Window | Positive Rate | Neg:Pos | ML Feasibility |
|---|---|---|---|---|---|
| Wound (H3) | 14 days | ~500 | ~0.41% | ~245:1 | ✅ Feasible (harder) |

### Monthly observation windows (~57,000 resident-months)

| Model Target | Horizon | Events in Window | Positive Rate | Neg:Pos | ML Feasibility |
|---|---|---|---|---|---|
| Elopement | 30 days | 7 | ~0.012% | ~8,000:1 | ❌ Impossible |

**Key insight:** Shorter horizons (7d vs 30d) increase imbalance ~4×. Falls and RTH remain learnable; wounds are harder but feasible. Altercations can be approached as resident-level classification (4.2% prevalence) rather than temporal window prediction. Med errors and elopement require business rules. Choking is present in raw incidents but excluded because it is not in the assignment's claim breakdown.

**Handling strategy (Tier 1 ML models):**
- LightGBM: `scale_pos_weight = neg_count / pos_count`
- Focal loss as alternative for extreme imbalance (wounds)
- Evaluate with **AUPRC** (more informative than AUROC under imbalance)
- Consider downsampling negatives for training speed (not for evaluation)

---

## 12. Business Impact Estimates

Computed as: `events_per_100_residents_per_year × avg_cost_per_event`

| Model | Avg Cost | Annual Exposure / 100 residents | Savings at 20% reduction |
|---|---|---|---|
| H1 — Fall | $3,500 | ~$286k | ~$57k |
| H2 — RTH | $20,000 | ~$757k | **~$151k** |
| H3 — Wound | $4,000 | ~$104k | ~$21k |
| H4 — Altercation | $2,500 | ~$25k | ~$5k |
| Rules — Med Error | $5,000 | ~$15k | ~$3k |
| Rules — Elopement | $2,500 | ~$1k | negligible |
| **Total** | | **~$1,188k** | **~$237k** |

> RTH is the clear priority for business impact. A 20% reduction in RTH events per 100 residents saves ~$151k/year — roughly 3× the value of equivalent fall reduction. The rule-based alerts for rare events add minimal $ but address liability/compliance concerns.

---

## 13. Modeling Plan Summary (Revised)

### Modeling Tiers

| Tier | Models | Approach | Events |
|---|---|---|---|
| **Tier 1 — Full ML** | H1 (Falls 7d), H2 (RTH 7d), H3 (Wounds 14d) | LightGBM binary classifiers, temporal CV, SHAP | 589–2,505 |
| **Tier 2 — Hybrid ML + Rules** | H4 (Altercations 7d) | Resident-level risk score (LR/GBM) + rule-based triggers | 228 (127 residents) |
| **Tier 3 — Business Rules Only** | Med Errors, Elopement | Rule-based flags from diagnoses, orders, document_tags | 7–44 |
| **Tier 4 — Composite Score** | All combined | Calibrated probabilities + rule flags → expected cost ranking | — |

### Why Not Group Rare Events Into One Model?

- Combined volume (279 modeled non-Tier-1 events) is still too sparse at weekly granularity
- Clinically heterogeneous: altercations (behavioral), med errors (pharmacological), elopement (cognitive)
- Different interventions needed — "something bad might happen" is not actionable
- A grouped model would be dominated by altercations (79% of group) and learn nothing about the other types

### H4 Altercations — Two-Stage Approach

**Stage 1 (ML):** Resident-level risk classification — "Is this resident prone to altercations?"
- 127 positive / 3,000 total = 4.2% positive rate → learnable
- Features: behavioral diagnoses, `aggressive_behavior` document_tags, psychotropic med needs, prior history

**Stage 2 (Rules):** Escalation triggers — "When should we alert?"
- New psychotropic medication order
- Medication refusal spike
- Pain level spike from vitals
- Facility-level altercation cluster

### Tier 3 — Business Rules

| Event | Key Rules | Data Sources |
|---|---|---|
| **Med Errors** | Polypharmacy (>15 concurrent), new pharmacy orders in 7d, cognitive impairment dx, "Psychotropic Medication Monitoring" need, prior med error, high missed/refused rate | `medications`, `physician_orders`, `diagnoses`, `needs`, `incidents` |
| **Elopement** | Dementia/Alzheimer's dx, wandering/elopement document_tags, prior elopement, cognitive ADL decline, new admission (first 30d) | `diagnoses`, `document_tags`, `incidents`, `adl_responses`, `residents` |

### Data Pipeline

```
Observation unit: resident × 7-day window (H1, H2), 14-day window (H3)
Observation stride: Non-overlapping (stride = horizon length)
Prediction gap (embargo): 1 day (features use data ≤ t-1d, label starts at t)

Feature groups (by coverage):
  HIGH  (~82%): vitals rolling features (3d, 7d, 14d lookback), diagnoses flags, history counts
  HIGH  (~83%): care plan needs by category
  MED   (~39%): lab report abnormal/critical counts
  MED   (~30%): RTH history, transfer outcomes
  MED   (~34%): document_tag semantic flags
  LOW   (~16%): medication adherence rates (facility subset)
  VERY LOW (3–4%): ADL/GG functional scores (recent cohort only)

Label creation:
  fall_7d   → any Fall incident in [t, t+7d]
  rth_7d    → any unplanned hospital_transfer in [t, t+7d]
  wound_14d → any Wound incident in [t, t+14d]
  altercation_resident → resident-level binary (ever had altercation)

Temporal split (3-fold expanding window CV + final hold-out):
  Fold 1: Train [Jul'23 – Mar'24] → Val [Apr'24 – Jun'24]
  Fold 2: Train [Jul'23 – Jun'24] → Val [Jul'24 – Sep'24]
  Fold 3: Train [Jul'23 – Sep'24] → Val [Oct'24 – Dec'24]
  Final:  Train [Jul'23 – Dec'24] → Test [Jan'25]
  Boundary embargo: 1 week between train end and val start
```

### Temporal Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Observation stride** | Non-overlapping (7d/14d) | Avoids label overlap and autocorrelation; clean independence |
| **Prediction gap** | 1 day | Accounts for data entry lag + intervention response time |
| **CV strategy** | Expanding window (3 folds) | Limited data (82 weeks) — expanding maximizes training samples |
| **Why not sliding window?** | Would discard oldest data | Older patterns (dx → falls) remain informative; rare events need volume |
| **Boundary handling** | 1-week embargo + purging | Prevents label leakage at train/val boundary |
| **Cold-start residents** | Diagnoses + demographics + facility base rate | LightGBM handles missing vitals natively; flag low-confidence |

### Feature Lookback Windows

| Feature Group | Lookback | Temporal Logic |
|---|---|---|
| Vitals (rolling stats) | 3d, 7d, 14d | mean, std, min, max per vital_type |
| Vitals (trend) | 7d, 14d | Linear regression slope on value vs. time |
| Diagnoses | Point-in-time | `onset_at ≤ t` AND (`resolved_at > t` OR null) |
| Incident history | 7d, 30d, 90d, all-time | Count by type in each lookback window |
| RTH history | 30d, 90d, all-time | Count of prior unplanned transfers |
| Medications | 7d, 14d | Missed rate, refused rate, active count |
| New orders | 7d | Count of new physician_orders (Pharmacy, Dietary) |
| Lab reports | 14d, 30d | Count of Abnormal + Critical results |
| Care needs | Point-in-time | Open needs by category at time t |
| ADL/GG | Latest before t | Most recent assessment score + change |
| Demographics | Static | Age at t, days_since_admission, facility_id |

### Production Inference & Retraining

| Model | Scoring Frequency | Retraining | Monitoring |
|---|---|---|---|
| H1: Falls (7d) | **Daily** | Monthly | AUPRC on 30d rolling, PSI on top features |
| H2: RTH (7d) | **Daily** | Monthly | AUPRC on 30d rolling, calibration drift |
| H3: Wounds (14d) | **Twice/week** | Monthly | AUPRC, prediction mean drift |
| H4: Altercations | **Weekly** | Quarterly | Precision at top-k |
| Tier 3: Rules | **Real-time** (on data events) | No retraining (quarterly review) | Precision/recall of rule set |

**Drift detection triggers (early retraining):**
- PSI > 0.2 on any top-10 feature
- AUPRC drops >15% relative to baseline
- Observed event rate shifts >30% from historical for 2+ consecutive weeks
- Calibration Hosmer-Lemeshow p < 0.05

### Implementation Order

| Priority | Component | Justification |
|---|---|---|
| 1 | H1: Falls (7d) | Most data, rich features, dual impact (incidents + RTH) |
| 2 | H2: RTH (7d) | Highest $ value, strong vital signs signal |
| 3 | Tier 3 rules | Zero ML needed — pure logic, immediate deployability |
| 4 | H3: Wounds (14d) | Moderate data, functional status + nutrition signal |
| 5 | H4: Altercations | Resident-level classification + rule triggers |
| 6 | Tier 4: Composite score | Requires all above; final integration |

### Notebook Roadmap

| Notebook | Purpose | Status |
|---|---|---|
| `01_eda.ipynb` | EDA + hypotheses + modeling strategy | ✅ Done |
| `02_feature_engineering.ipynb` | Feature matrix construction (7d/14d windows) | 🔲 Next |
| `03_modeling_fall.ipynb` | H1: Fall model (7d horizon) | 🔲 Planned |
| `04_modeling_rth.ipynb` | H2: RTH model (7d horizon) | 🔲 Planned |
| `05_modeling_wound.ipynb` | H3: Wound model (14d horizon) | 🔲 Planned |
| `06_business_rules.ipynb` | Tier 3 rules + H4 altercation classifier | 🔲 Planned |
| `07_composite_score.ipynb` | Tier 4: Composite score + business report | 🔲 Planned |
