# Thesis Experiments — Family Run Results Summary

**Date:** 2026-11-06 · **Scripts:** `experiments/full_pipeline_family.py` (pilot),
`experiments/full_pipeline_combined.py` (main) · **Logs:**
`results_family_run.log`, `results_family_combined_run.log`

**How this was verified (thorough check, as instructed):** every number below was
recomputed independently from the result CSVs (not copied from the console log):
weighted accuracy from `window_accuracy.csv`, stability means from
`cycle_stability.csv`, drift rows from `drifts.csv`, per-snapshot accuracy by
joining each snapshot's `end_row` to its containing window, and top-feature
lists from `cycle_explanations.csv`. Downloaded file sizes were checked against
the HuggingFace mirror listing (all exact), and row counts / 39-column match
were verified for every file before running. Log output and CSV recomputation
agree everywhere (the only difference is rounding: the log prints river
Accuracy as 1.0000, the recomputed weighted accuracy is 0.999997). **No re-run
was needed — nothing looked off.**

## What ran

Two runs, same pipeline (Hoeffding Tree + prequential evaluation + ADWIN
delta=0.002 + reset-on-drift retraining + TreeSHAP/LIME on a surrogate
DecisionTree + Kendall's tau / top-7 Jaccard), binary labels (Benign=0,
Attack=1), 10,000-row windows, checkpoints every 100,000 rows:

| Run | Stream (file order = time order) | Rows | Windows |
|---|---|---|---|
| **Pilot** | Benign → 6 DDoS floods (TCP, UDP, SYN, ICMP, HTTP, PSHACK) | 1,725,377 | 173 |
| **Combined (main)** | Benign → **Phase A: 8 DDoS floods** (TCP, UDP, SYN, ICMP, HTTP, PSHACK, RSTFIN, SynonymousIP) → **Phase B: 5 Recon** (PortScan, PingSweep, OSScan, HostDiscovery, VulnerabilityScan) | 2,951,471 | 296 |

Phase A ends at row 2,260,936; Phase B is rows 2,260,937–2,951,470 (690,534 rows).
Fixed explanation sample: 30 instances (pilot) / 63 instances (combined);
39 numeric features throughout; 0 missing values.

## Pilot (6 floods) — headline numbers

- Weighted accuracy **0.999994**; only 2 windows < 1.0 (window 1: 0.9999 warm-up;
  window 37: 0.9991, the Benign→DDoS switch).
- **1 drift** (ADWIN) at row 362,431 — the Benign→DDoS switch. No drift at any
  flood→flood switch.
- 16 snapshots explained (ids 3–18), 15 transitions per method.
- **SHAP: mean τ 0.412, mean top-7 Jaccard 0.765** (min τ −0.053, min Jaccard 0.556).
- **LIME: mean τ 0.744, mean top-7 Jaccard 0.370** (min τ +0.592, min Jaccard 0.077).
- **No collapse within the flood family:** SHAP Jaccard never below 0.556.

## Combined run (Phase A + Phase B) — headline numbers

- Weighted accuracy **0.999997**; only 2 windows < 1.0 (window 1: 0.9999;
  window 37: 0.9991 — the same two as run 1 and the pilot).
- **1 drift** (ADWIN) at row 362,431. **No drift at the flood→Recon switch**
  (row 2,260,937) and none at any flood→flood switch.
- 28 snapshots explained (ids 3–30), 27 transitions per method; 3 skipped
  (checkpoints 0–2, benign-only buffers). Runtime 293.9 s.
- **Overall: SHAP τ 0.384 / Jaccard 0.644; LIME τ 0.667 / Jaccard 0.359.**

### Phase A vs Phase B (combined run)

| Segment | Transitions | SHAP τ | SHAP Jaccard | LIME τ | LIME Jaccard |
|---|---|---|---|---|---|
| Phase A (8 floods) | 19 | 0.320 | **0.762** | 0.695 | 0.358 |
| Phase B (5 Recon) | 8 | 0.537 | 0.365 | 0.599 | 0.362 |

### Phase B — the three questions

1. **Did ADWIN detect the drift when the new pattern arrived? NO.**
   No drift fired at the flood→Recon switch (row 2,260,937) or anywhere in
   Phase B. The only drift in the entire 2.95M-row run is at row 362,431
   (Benign→DDoS). ADWIN watches the *error* stream, and there were no errors
   to watch — see Q2.
2. **After retraining, does the model recognise the new attack? YES,
   immediately.** Every window around and after the switch (windows 226–241)
   has accuracy **1.0**. The model never failed on Recon traffic, so no
   retraining was even triggered.
3. **Stability across the pattern change? It drops, then settles in a new
   regime.** First snapshot after the switch (snapshot 23, row 2,299,999):
   SHAP τ +0.049 / Jaccard 0.167; LIME τ +0.533 / Jaccard 0.167. After that,
   SHAP τ vs first stays ≈ **−0.183 for all of Phase B** and Jaccard vs first
   is 0.000–0.167 — the explanations moved to a new regime and never returned.
   Top features confirm it: Phase A SHAP top-7 is packet/flag features
   (`syn_flag_number`, `Header_Length`, `Time_To_Live`, `Protocol Type`,
   `Rate`, `fin_flag_number`, `rst_flag_number`); Phase B SHAP top-7 is
   protocol/rate features (`Max`, `HTTPS`, `HTTP`, `IAT`, `Rate`, `Tot size`).

### RQ3 — accuracy vs stability, snapshot by snapshot

Accuracy at **every one of the 28 snapshots is 1.0000** (joined from
`window_accuracy.csv`), including snapshot 23 (the pattern-change snapshot)
and all Phase B snapshots. **There is zero variation in accuracy to correlate
with stability, so RQ3 remains UNANSWERED — explicitly.** Stability fell
(SHAP Jaccard 0.167 at the switch; vs-first Jaccard 0.000 at snapshots 24
and 30) while accuracy stayed at ceiling. This is the same ceiling effect as
run 1, now confirmed on 2.95M rows and 13 attack files.

**Harder setup needed for RQ3 (to be done next):** (1) **multiclass**
classification (34 classes: benign + 33 attacks) instead of binary, so a new
attack family actually confuses the classifier and creates accuracy variation;
(2) **feature-based drift detection** (e.g. ADWIN on feature distributions /
KS-tests) instead of error-based ADWIN, so pattern changes are flagged even
when accuracy stays 1.0; (3) harder attack mixes (attacks overlapping benign
traffic, e.g. SlowLoris / web attacks) or train-on-fewer-families /
test-on-unseen-family splits; (4) the merged CICIoT2023 CSV with real
timestamps for true time-ordered windows; (5) finer checkpoints around switches
(e.g. every 10k rows) to catch any transient accuracy dip.

## RQ1 / RQ2 — updated answers

- **RQ1 (stability change across adaptation): YES, and it is regime-dependent
  — the collapse is a CROSS-FAMILY phenomenon, not a within-family one.**
  Within the flood family (Phase A + pilot) SHAP Jaccard never drops below
  0.556 and LIME τ never goes negative. Across the family switch (Phase B)
  stability drops sharply and never recovers to the initial ranking.
  Honest caveat: ADWIN fired only **once** in the whole run, so "adaptation
  cycles" here are the 28 checkpoints, not multiple ADWIN drift events —
  the requirement for multiple drift events was NOT met in ADWIN terms,
  because error-based detection cannot fire when there are no errors.
- **RQ2 (SHAP vs LIME): the run-1 pattern holds and is now based on 27
  transitions instead of 5.** SHAP's top-7 sets are more stable
  (Jaccard 0.644 vs LIME 0.359 overall; 0.762 vs 0.358 in Phase A), while
  LIME's full ranking is more consistent (τ 0.667 vs SHAP 0.384; LIME τ never
  negative, SHAP swings +1.000 to −0.053). One observed quirk: in Phase A,
  SHAP τ flips between +1.000 and ≈ −0.026 while Jaccard stays 0.556–1.000 —
  the top-7 set is stable but the full 39-feature ranking flips between two
  near-tied orderings (likely near-ties among low-importance tail features);
  Jaccard is the more robust signal for top features.

## Comparison with run 1

| Measure | Run 1 (3 files) | Pilot (6 floods) | Combined (8 floods + 5 Recon) |
|---|---|---|---|
| Rows | 710,421 | 1,725,377 | 2,951,471 |
| Windows | 72 | 173 | 296 |
| Weighted accuracy | 0.999986 | 0.999994 | 0.999997 |
| ADWIN drifts | 1 (362,431) | 1 (362,431) | 1 (362,431) |
| Snapshots / transitions | 6 / 5 | 16 / 15 | 28 / 27 |
| SHAP τ / Jaccard | 0.446 / 0.483 | 0.412 / **0.765** | 0.384 / 0.644 |
| LIME τ / Jaccard | 0.603 / 0.294 | 0.744 / 0.370 | 0.667 / 0.359 |
| Worst stability break | SHAP Jaccard 0.000, τ −0.204 (Recon switch) | none (min Jaccard 0.556) | SHAP Jaccard 0.167 at switch; vs-first 0.000 in Phase B |

## Thesis Objectives (Synopsis §6) — status against the numbers

1. **Adaptive IDS pipeline that detects and adapts to concept drift —
   PARTIALLY demonstrated.** Pipeline built and ran on 2,951,471 rows;
   ADWIN detected 1 drift (row 362,431) and retrained (cycle 0 → 1). BUT it
   did **not** detect the flood→Recon pattern change (no drift at row
   2,260,937), because accuracy never dropped. Error-visible drift is
   detected; error-invisible pattern change is not.
2. **Quantitative stability metrics (Kendall's τ, top-k Jaccard) applied
   across adaptation cycles — FULLY demonstrated.** Measured at all 28
   snapshots × 2 methods = 54 stability measurements, plus 30 in the pilot.
3. **Compare multiple XAI methods under drift AND analyse their relationship
   with detection performance — comparison DONE, relationship UNANSWERED.**
   SHAP vs LIME compared across 27 transitions (Phase A vs B split included).
   The relationship part could not be measured: accuracy is 1.0000 at every
   snapshot, so there is nothing to correlate stability against. See the
   harder setup listed under RQ3 above.

## Limitations (carry into the thesis)

1. Binary labels only (Benign vs Attack); multiclass not attempted.
2. 39 features, not 46 (same as run 1 — unresolved).
3. File order = time order (no timestamps in per-attack files).
4. Surrogate explanations (sklearn DecisionTree on buffer, not the Hoeffding
   tree itself).
5. Only 1 ADWIN drift in 2.95M rows — error-based detection is blind when
   accuracy is at ceiling.
6. RQ3 still unanswered (ceiling effect) — needs the harder setup above.
7. HuggingFace mirror, not the official CIC download (needs Ammara's form).
8. PingSweep file is tiny (2,262 rows); HTTP_Flood is small (28,790 rows) —
   they contribute few windows/snapshots.

## Files

- `experiments/full_pipeline_family.py` — pilot script (6 floods)
- `experiments/full_pipeline_combined.py` — main script (8 floods + 5 Recon)
- `experiments/results_family/pilot6/` — pilot CSVs + run log
- `experiments/results_family/combined/` — `window_accuracy.csv`,
  `drifts.csv`, `cycle_stability.csv`, `cycle_explanations.csv`
- `experiments/results_family_combined_run.log` — combined run console log
- `experiments/results_family/RESULTS_SUMMARY_FAMILY.md` — this file
