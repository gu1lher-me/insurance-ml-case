# Composite Risk Score Backtest Results

For the full technical modeling strategy behind these results, see
`docs/modeling-strategy.md`.

Scoring period: 2025-01-01 to 2025-02-01

The backtest estimates financial value by asking which observed holdout events
would have been preceded by an alert. Estimated savings apply the pilot
intervention-effectiveness assumptions in `modeling/business_policy.py`.

## Primary Policy

Policy: `top_10pct_per_facility`

- Alerts: 401 of 3,612 scored windows (11.1%)
- Actual claim exposure captured: $291,000 of $1,791,000 (16.2%)
- Estimated avoided claim dollars: $52,075
- Intervention cost: $40,100
- Estimated net savings: $11,975
- Estimated ROI: 0.30x

## Sensitivity

| Policy | Alerts | Captured claim cost | Avoided claim cost | Intervention cost | Net savings | ROI |
|---|---:|---:|---:|---:|---:|---:|
| `top_05pct_per_facility` | 226 | $168,000 | $30,600 | $22,600 | $8,000 | 0.35x |
| `top_10pct_per_facility` | 401 | $291,000 | $52,075 | $40,100 | $11,975 | 0.30x |
| `top_15pct_per_facility` | 590 | $448,000 | $80,475 | $59,000 | $21,475 | 0.36x |
| `top_20pct_per_facility` | 753 | $581,500 | $103,175 | $75,300 | $27,875 | 0.37x |
| `economic_threshold` | 435 | $350,500 | $63,975 | $43,500 | $20,475 | 0.47x |

## Primary Policy By Incident Type

| Incident type | Actual events | Captured events | Captured claim cost | Capture rate |
|---|---:|---:|---:|---:|
| Return to hospital | 54 | 6 | $120,000 | 11.1% |
| Fall | 159 | 39 | $136,500 | 24.5% |
| Wound / pressure injury | 33 | 8 | $32,000 | 24.2% |
| Altercation | 6 | 1 | $2,500 | 16.7% |
| Medication error | 1 | 0 | $0 | 0.0% |
| Choking | 1 | 0 | $0 | 0.0% |

## Interpretation

These are backtested financial indicators, not causal proof. The strongest
observed metric is captured claim exposure: dollars from events that had an
alert before they occurred. Net savings depends on the intervention cost and
effectiveness assumptions and should be validated in a prospective pilot.
