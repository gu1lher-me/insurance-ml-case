# Composite Risk Score Backtest Results

For the full technical modeling strategy behind these results, see
`docs/modeling-strategy.md`.

Scoring period: 2025-01-01 to 2025-02-01

The backtest estimates financial value by asking which observed holdout events
would have been preceded by an alert. Estimated savings apply one uniform
scenario assumption for intervention effectiveness, defined in
`modeling/business_policy.py`.

Baseline intervention effectiveness: 20.0%

## Primary Policy

Policy: `top_10pct_per_facility`

- Alerts: 401 of 3,612 scored windows (11.1%)
- Actual claim exposure captured: $304,500 of $1,788,500 (17.0%)
- Estimated avoided claim dollars: $60,900
- Intervention cost: $40,100
- Estimated net savings: $20,800
- Estimated ROI: 0.52x

## Sensitivity

### Policy Sensitivity

| Policy | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---|---:|---:|---:|---:|---:|---:|
| `top_05pct_per_facility` | 226 | $192,000 | $38,400 | $22,600 | $15,800 | 0.70x |
| `top_10pct_per_facility` | 401 | $304,500 | $60,900 | $40,100 | $20,800 | 0.52x |
| `top_15pct_per_facility` | 590 | $449,500 | $89,900 | $59,000 | $30,900 | 0.52x |
| `top_20pct_per_facility` | 753 | $584,500 | $116,900 | $75,300 | $41,600 | 0.55x |
| `economic_threshold` | 455 | $334,000 | $66,800 | $45,500 | $21,300 | 0.47x |

### Primary Policy Effectiveness Sensitivity

| Assumed effectiveness | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0% | 401 | $304,500 | $30,450 | $40,100 | -$9,650 | -0.24x |
| 15.0% | 401 | $304,500 | $45,675 | $40,100 | $5,575 | 0.14x |
| 20.0% | 401 | $304,500 | $60,900 | $40,100 | $20,800 | 0.52x |
| 25.0% | 401 | $304,500 | $76,125 | $40,100 | $36,025 | 0.90x |

## Primary Policy By Incident Type

| Incident type | Actual events | Captured events | Captured claim cost | Capture rate |
|---|---:|---:|---:|---:|
| Return to hospital | 54 | 7 | $140,000 | 13.0% |
| Fall | 159 | 39 | $136,500 | 24.5% |
| Wound / pressure injury | 33 | 7 | $28,000 | 21.2% |
| Altercation | 6 | 0 | $0 | 0.0% |
| Medication error | 1 | 0 | $0 | 0.0% |

## Interpretation

These are backtested financial indicators, not causal proof. The strongest
observed metric is captured claim exposure: dollars from events that had an
alert before they occurred. Net savings depends on intervention cost and
the uniform effectiveness scenario, which should be replaced with measured
effects from a prospective pilot.
