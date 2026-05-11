# ML & Data Science: Take-Home Assignment

## Background

Tricura Insurance Group provides liability insurance to skilled nursing facilities. A significant portion of our claims exposure comes from incidents involving residents.

Claims break down roughly as follows:

| Incident Type | % of Claims | Avg Cost per Incident |
|---|---|---|
| Falls | ~13% | ~$3,500 |
| Medication errors | ~10% | ~$5,000 |
| Wounds / pressure injuries | ~7% | ~$4,000 |
| Return-to-hospital (RTH) events | ~7% | ~$20,000 |
| Elopement / wandering | ~5% | ~$2,500 |
| Altercations | ~2% | ~$2,500 |

## Business Challenge

Our margins depend on keeping actual claims below the premiums we collect. Raising premiums isn't sustainable without losing clients, so the lever we're exploring is reducing the frequency and severity of incidents at the facilities we insure.

## Goal

Build a working model that addresses this business challenge. You decide what to predict, how to frame the problem, which data to use, and how to validate it.

## Data

The `data/` folder contains anonymized records from our backend system, covering a sample of residents across skilled nursing facilities.

## Deliverable

- A public git repository with your code (notebooks, scripts, or both) and README.md writeup: what you built, why, key findings
- Be prepared to walk through your decisions in a live conversation
