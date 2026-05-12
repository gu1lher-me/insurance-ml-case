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
- Actual claim exposure captured: $399,500 of $1,788,500 (22.3%)
- Estimated avoided claim dollars: $79,900
- Intervention cost: $40,100
- Estimated net savings: $39,800
- Estimated ROI: 0.99x

## Sensitivity

### Policy Sensitivity

| Policy | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---|---:|---:|---:|---:|---:|---:|
| `top_05pct_per_facility` | 226 | $212,500 | $42,500 | $22,600 | $19,900 | 0.88x |
| `top_10pct_per_facility` | 401 | $399,500 | $79,900 | $40,100 | $39,800 | 0.99x |
| `top_15pct_per_facility` | 590 | $474,500 | $94,900 | $59,000 | $35,900 | 0.61x |
| `top_20pct_per_facility` | 753 | $557,500 | $111,500 | $75,300 | $36,200 | 0.48x |
| `economic_threshold` | 995 | $795,500 | $159,100 | $99,500 | $59,600 | 0.60x |

### Primary Policy Effectiveness Sensitivity

| Assumed effectiveness | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 15.0% | 401 | $399,500 | $59,925 | $40,100 | $19,825 | 0.49x |
| 20.0% | 401 | $399,500 | $79,900 | $40,100 | $39,800 | 0.99x |
| 25.0% | 401 | $399,500 | $99,875 | $40,100 | $59,775 | 1.49x |

## Primary Policy By Incident Type

| Incident type | Actual events | Captured events | Captured claim cost | Capture rate |
|---|---:|---:|---:|---:|
| Return to hospital | 54 | 11 | $220,000 | 20.4% |
| Fall | 159 | 41 | $143,500 | 25.8% |
| Wound / pressure injury | 33 | 9 | $36,000 | 27.3% |
| Altercation | 6 | 0 | $0 | 0.0% |
| Medication error | 1 | 0 | $0 | 0.0% |

## Interpretation

These are backtested financial indicators, not causal proof. The strongest
observed metric is captured claim exposure: dollars from events that had an
alert before they occurred. Net savings depends on intervention cost and
the uniform effectiveness scenario, which should be replaced with measured
effects from a prospective pilot.
