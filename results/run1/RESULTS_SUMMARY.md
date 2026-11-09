# Thesis Experiments — Results Summary

**Date:** 2026-11-06 · **Script:** `experiments/full_pipeline.py` · **Log:** `results/full_pipeline_log.txt`

## Setup (what actually ran)

- **Data:** 3 local CICIoT2023 files, streamed in order as time windows of 10,000 rows:
  BenignTraffic (362,361 rows, label 0) → DDoS-TCP_Flood (265,776, label 1) →
  Recon-PortScan (82,284, label 1). **Total: 710,421 rows, 39 numeric features.**
  Files have no timestamp column, so **file order is the time order**
  (documented assumption). Labels come from file names (Benign = 0, Attack = 1).
- **Model:** river `HoeffdingTreeClassifier` (grace_period=200, delta=1e-5,
  leaf_prediction="nb"), prequential evaluation (predict → score → learn).
- **Drift:** river `ADWIN` (delta=0.002) on the prediction-error stream.
  **Retrain on drift = discard the tree and ADWIN and start fresh** on the
  post-drift stream.
- **Explanations:** fixed sample of 30 instances (10 per file, fixed row
  positions) explained at every snapshot. TreeSHAP + LIME explain a
  **surrogate sklearn DecisionTree** fitted to the current buffer (last 5,000
  benign + last 5,000 attack rows) — river's Hoeffding tree has no native
  SHAP/LIME support, so this is the documented surrogate choice.
  LIME: 500 samples per instance, top 10 features. Stability: Kendall's tau
  on mean-|importance| ranking vectors, top-7 Jaccard on top-7 feature sets.
- **Data cleaning:** the `Rate` feature contains infinite values (19 Benign,
  9 DDoS, 1 Recon). `+-inf` was replaced by the column median of the buffer
  before fitting the surrogate (river's tree accepts inf; sklearn rejects it).
- **Runtime:** ~56 seconds for the full stream + explanations.

## Headline numbers

| Measure | Value |
|---|---|
| Rows processed | 710,421 |
| Overall accuracy | 1.0000 (mean window accuracy 0.999986) |
| Windows (10k rows) | 72 |
| Windows below 1.0 | 2 (window 1: 0.9999 warm-up; window 37: 0.9991) |
| Drifts detected (ADWIN) | **1** — row 362,431 (70 rows after the Benign→DDoS switch) |
| Explanation snapshots | **6** (1 drift + 4 checkpoints + 1 final) |
| Snapshots skipped | 3 (checkpoints at rows 99,999 / 199,999 / 299,999 — buffer held a single class, benign only) |

**The DDoS→Recon switch (row 628,137) did NOT trigger a drift** — the error
rate never moved enough for ADWIN. The model adapted within ~70 rows of the
first switch and then essentially never erred again.

## Explanation stability across snapshots (RQ1 + RQ2)

Snapshots: 3 = drift (row 362,431) · 4 = checkpoint (399,999) · 5 = checkpoint
(499,999) · 6 = checkpoint (599,999) · 7 = checkpoint (699,999) · 8 = final
(710,420). "vs prev" = previous snapshot; "vs first" = snapshot 3.

| Snapshot | Transition | Method | Kendall τ vs prev | Top-7 Jaccard vs prev | τ vs first | Jaccard vs first |
|---|---|---|---|---|---|---|
| 4 | drift→checkpoint | SHAP | +0.712 | 0.750 | +0.712 | 0.750 |
| 4 | drift→checkpoint | LIME | +0.538 | 0.273 | +0.538 | 0.273 |
| 5 | checkpoint→checkpoint | SHAP | +0.395 | 0.556 | +0.585 | 0.556 |
| 5 | checkpoint→checkpoint | LIME | +0.682 | 0.167 | +0.615 | 0.273 |
| 6 | checkpoint→checkpoint | SHAP | +0.585 | 0.556 | **+1.000** | **1.000** |
| 6 | checkpoint→checkpoint | LIME | +0.703 | 0.400 | +0.517 | 0.167 |
| 7 | checkpoint→checkpoint | SHAP | **−0.204** | **0.000** | −0.204 | 0.000 |
| 7 | checkpoint→checkpoint | LIME | +0.401 | 0.077 | +0.336 | 0.273 |
| 8 | checkpoint→final | SHAP | +0.744 | 0.556 | −0.199 | 0.167 |
| 8 | checkpoint→final | LIME | +0.693 | 0.556 | +0.375 | 0.167 |

**Means (vs previous snapshot):**

| Method | Mean Kendall τ | Mean top-7 Jaccard |
|---|---|---|
| SHAP | 0.446 | **0.483** |
| LIME | **0.603** | 0.294 |

### RQ1 — Does explanation stability change as the model updates?
**Yes, and it is regime-dependent.** Inside the DDoS phase (snapshots 3→6)
explanations are moderately stable (SHAP τ 0.40–0.71, Jaccard 0.56–0.75;
snapshot 6 reproduces snapshot 3's SHAP ranking exactly, τ = +1.000).
At the Recon transition (snapshot 7) stability collapses: SHAP τ turns
negative (−0.204) and its top-7 has **zero overlap** with the previous
snapshot (Jaccard 0.000); LIME also drops to its lowest Jaccard (0.077).
The final model (snapshot 8) never returns to the initial ranking
(SHAP τ vs first −0.199, Jaccard 0.167).

### RQ2 — SHAP vs LIME stability across adaptation cycles?
**Nuanced, not one-sided.** SHAP's **top-7 feature sets are more stable**
(mean Jaccard 0.483 vs LIME 0.294; SHAP never drops below 0.556 except at
the Recon break). LIME's **full ranking is more consistent** (mean τ 0.603
vs SHAP 0.446; LIME τ never goes negative, range 0.40–0.70, while SHAP
swings from +0.74 to −0.20). In short: SHAP keeps the same top features
more often, but when it changes, it changes hard; LIME's top-7 churns
more, but its overall ranking moves less violently.

### RQ3 — Does explanation instability coincide with detection degradation?
**No degradation was observed.** Accuracy stayed at 1.0 in every window
except the warm-up (window 1: 0.9999) and the drift window (window 37:
0.9991, recovering to 1.0 in window 38). The biggest explanation break
(snapshot 7, Recon transition) coincides with **no accuracy drop at all**
(windows 69–72: all 1.0). So in this run, explanations moved substantially
while detection stayed at ceiling. **Caveat:** accuracy is at ceiling
(mean 0.999986), so there is almost no room to observe degradation — this
is a ceiling effect, not proof that instability can never hurt detection.

## Top features (for Chapter 4)

- Early/DDoS phase top features (SHAP): `Number`, `Header_Length`,
  `Time_To_Live`, `Protocol Type`, `fin_flag_number`, `syn_flag_number`,
  `rst_flag_number`.
- Late/Recon phase top features (SHAP): `HTTP`, `HTTPS`, `Max`, `Rate`,
  `psh_flag_number`, `Time_To_Live`, `Header_Length`.
- The shift from packet/flag features to protocol/rate features is exactly
  what the stability numbers measure (see snapshot 7 collapse).

## Limitations — must be stated in the thesis

1. **Only 3 of 33 attack files used** (Benign, DDoS-TCP_Flood,
   Recon-PortScan). The full dataset has 33 attacks; the other 30 files
   were not downloaded (full mirror ~8.9 GB).
2. **Binary labels only** (Benign vs Attack). Multiclass (34 classes) was
   not attempted.
3. **39 features, not 46.** Literature says 46 features + label; the local
   per-attack CSVs have 39 columns. The missing ones (flow_duration,
   Duration, Srate, Drate, Magnitude, Radius, Covariance, Weight, …) are
   not in these files. Must be resolved against the full merged CSV before
   Chapter 3 is finalised.
4. **File order = time order.** No timestamps in the per-attack files, so
   real time-ordered windows need the merged dataset.
5. **Surrogate explanations.** SHAP/LIME explain a sklearn DecisionTree
   fitted to the current window buffer, not the Hoeffding tree itself.
6. **Only 1 drift detected.** Error-based drift detection found almost
   nothing because the binary task on these three files is near-trivially
   separable (accuracy ≈ 1.0). RQ3 therefore has a ceiling effect.
7. **Dataset provenance:** local files come from the HuggingFace mirror
   `bencorn/CIC-IoT-2023`, not the official CIC download (which needs a
   form with Ammara's details). For the final thesis, download from the
   official CIC page as well.
8. **6 snapshots only** (5 transitions). The stability conclusions rest on
   a small number of comparisons — report the numbers, not over-claim.

## Files

- `experiments/full_pipeline.py` — the pipeline (readable, commented)
- `experiments/results/window_accuracy.csv` — accuracy per 10k window (72 rows)
- `experiments/results/drifts.csv` — ADWIN drift positions (1 row)
- `experiments/results/cycle_stability.csv` — Kendall τ + top-7 Jaccard per snapshot/method (10 rows)
- `experiments/results/cycle_explanations.csv` — full 39-feature rankings per snapshot/method (468 rows)
- `experiments/results/full_pipeline_log.txt` — console log of the run
- `experiments/results/RESULTS_SUMMARY.md` — this file
