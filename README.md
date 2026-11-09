# Evaluating Explanation Stability in Concept-Drift-Adaptive Intrusion Detection for Lightweight IoT Devices

MS thesis research — Ammara Ali Ahmed, Department of Computer Science, University of Okara (2025–2027). Supervisor: Ms. Ifrah Pervaz.

## The question

IoT attacks keep changing (concept drift), so an intrusion detection system must keep retraining itself (adaptive IDS). But when the model updates, do its explanations (SHAP / LIME) stay stable — or do they change? Nobody had measured that. This thesis measures it.

## What's in this repo (so far)

- `synopsis/Research_Synopsis_Updated_Oct2026.docx` — the approved research synopsis (submitted to the department, 7 October 2026).
- `abstract/ICAIS2026_Abstract_AmmaraAliAhmed.docx` — conference abstract summarising the experiments and headline results (November 2026).
- `code/` — the experiment pipelines (Python: Hoeffding Tree + ADWIN drift detection, TreeSHAP + LIME, Kendall's τ and top-k Jaccard stability metrics) and `requirements.txt`.
- `results/` — result files for all three runs: `run1/` (binary, 3 files), `family/` (binary, all 13 files: benign + 8 DDoS floods + 5 Recon), `rq3/` (multiclass, 14 classes) — stability tables, window accuracy, drift logs and summaries.

## Headline results (from the abstract)

- Dataset: CICIoT2023 (2,951,471 flows: benign traffic + 8 DDoS floods + 5 Recon scans).
- Same-family attacks: SHAP top-7 Jaccard 0.762 (stable). At the switch to a new attack family: 0.167 (collapse). After adaptation: 0.365 (settles in a new regime).
- Detection accuracy stayed 99.99% throughout — explanation instability did not degrade detection (RQ3: negative result).

## Status

Work in progress. The full thesis document will be added once it is finalised (submission: May 2027). The dataset (CICIoT2023, HuggingFace mirror `bencorn/CIC-IoT-2023`) is not included — see `code/SETUP_NOTES.md` for how to fetch it.
