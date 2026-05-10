# Insurance ML Case

This repository contains a tiered incident-risk modeling system for skilled
nursing liability claims.

Key docs:

- `docs/modeling-strategy.md` - complete technical modeling and financial strategy
- `docs/backtest-results.md` - Jan 2025 composite-score financial backtest
- `docs/prediction-pipeline.md` - new-batch prediction and daily scoring workflow

Run daily scoring:

```powershell
.venv\Scripts\python.exe pipelines\run_daily_scoring.py --as-of-date 2025-02-01
```
