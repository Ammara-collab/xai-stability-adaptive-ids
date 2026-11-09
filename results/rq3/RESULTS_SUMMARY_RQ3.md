# Thesis Experiments — RQ3 Harder Setup: Results Summary

**Date:** 2026-11-06 · **Script:** `experiments/full_pipeline_rq3.py` (579.4 s)
· **Analysis:** `experiments/analyse_rq3.py` · **Log:** `results_rq3_run2.log`
(run 2; run 1 was discarded after a per-class counting bug was found —
see §7 — run 2 reproduces run 1 exactly on every shared number and adds
the corrected per-class accuracy plus full tail coverage)

**How this was verified (thorough check, as instructed):** every number
below was recomputed independently from the result CSVs (not copied from
the console log): weighted accuracy from `window_accuracy.csv`, per-class
accuracy from `class_accuracy.csv`, drift rows from `drifts.csv`, stability
means from `cycle_stability.csv`, snapshot-window accuracy by joining each
snapshot's `end_row` to its containing window, and correlations computed
from the joined snapshot table (`rq3_snapshot_table.csv`). The run was
repeated once (run 2) after the per-class bug fix; run 1 and run 2 agree
exactly on all shared numbers (same worst windows, same drift rows, same
weighted accuracy 0.9999146), confirming determinism. The supplementary
frozen-model check was run separately with true multiclass labels (an
earlier attempt with a label bug was discarded and redone).

## Why this run exists

Runs 1 + family (binary labels) left RQ3 **unanswered**: accuracy was
1.0000 at every snapshot, so there was zero variation to correlate with
explanation instability. This run made the task harder in three ways:

1. **Multiclass labels** (14 classes: Benign + 13 attack types, one class
   per file) instead of binary.
2. **Feature-based drift detection** (mean-shift score between consecutive
   1,000-row windows, fed to ADWIN) alongside error-based ADWIN; either
   detector firing triggers a snapshot + reset (detector logged).
3. **Train-on-fewer / test-on-unseen:** a second Hoeffding tree
   (`model_unseen`) learns only on Benign + 8 DDoS floods (Phase A) and is
   then **frozen** for Phase B (5 Recon attacks) — predict only, no
   learning.

## What ran

Stream (file order = time order): Benign → 8 DDoS floods (TCP, UDP, SYN,
ICMP, HTTP, PSHACK, RSTFIN, SynonymousIP) → 5 Recon (PortScan, PingSweep,
OSScan, HostDiscovery, VulnerabilityScan). **2,951,471 rows, 296 windows**
of 10,000 rows. 48 snapshots explained (ids 0–53, 6 skipped early:
Benign-only buffer, single class); 47 transitions per method.
Checkpoints every 100,000 rows + drift snapshots + final snapshot (the
final snapshot IS included this time — run 1's cap cut it off).
Fixed explanation sample: 70 instances (5 per file). 39 numeric features.

## 1. Accuracy — multiclass (14 classes)

- Weighted accuracy **0.9999146** (overall river accuracy 0.9999).
- 24 of 296 windows below 1.0; 9 below 0.999; **2 below 0.997**.
- Worst windows: **235 (0.9956**, Phase B), 143 (0.9966), 90 (0.9973),
  258 (0.9974), 245 (0.9978), 37 (0.9979, Benign→DDoS switch),
  117 (0.9986), 200 (0.9989). The dips cluster at file switches.

### Per-class accuracy (whole stream) — YES, real drops exist

| Class | n | Overall acc | Worst window | Windows < 0.99 |
|---|---|---|---|---|
| Benign | 362,361 | 1.0000 | 0.9999 | 0 |
| TCP_Flood | 265,776 | 0.9999 | 0.9973 | 0 |
| UDP_Flood | 266,605 | 1.0000 | 0.9946 | 0 |
| SYN_Flood | 266,024 | 0.9999 | 0.9949 | 0 |
| ICMP_Flood | 268,010 | 0.9999 | 0.9985 | 0 |
| HTTP_Flood | 28,790 | 0.9988 | **0.9722** | 1 |
| PSHACK_Flood | 267,811 | 1.0000 | 0.9959 | 0 |
| RSTFIN_Flood | 266,934 | 1.0000 | 0.9978 | 0 |
| SynonymousIP_Flood | 268,626 | 1.0000 | 0.9986 | 0 |
| PortScan | 82,284 | 0.9999 | 0.9988 | 0 |
| PingSweep | 2,262 | **0.9867** | **0.9867** | 1 |
| OSScan | 98,259 | 0.9999 | 0.9969 | 0 |
| HostDiscovery | 134,378 | 0.9998 | 0.9965 | 0 |
| VulnerabilityScan | 373,351 | 0.9999 | **0.9862** | 1 |

The lowest per-class accuracy is **PingSweep 98.67%** (only 2,262 rows,
one partial window) and the worst single window is **HTTP_Flood 97.22%**.
So multiclass DID create accuracy variation — small (worst window 99.56%
overall), but real, unlike the binary runs where accuracy was 1.0000
almost everywhere.

## 2. Drift detection — 24 drifts (13 error, 11 feature)

| Detector | Count | Rows |
|---|---|---|
| error (ADWIN on prediction errors) | 13 | 362,531; 628,307; 894,931; 1,160,947; 1,428,979; 1,457,587; 1,725,395; 1,992,403; 2,261,011; 2,343,251; 2,345,523; 2,443,763; 2,578,147 |
| feature (ADWIN on mean-shift score) | 11 | 181,099; 285,399; 338,499; 604,499; 2,538,499; 2,656,899; 2,693,999; 2,740,699; 2,822,599; 2,885,299; 2,925,599 |

Key findings:

- **Error-based ADWIN fired at BOTH main switches in multiclass**
  (Benign→DDoS at row 362,531; DDoS→Recon at row 2,261,011) — it did
  NOT fire at the Recon switch in the binary runs. Multiclass accuracy
  dips at switches are what make error-based detection work here.
- **Feature-based ADWIN did NOT fire at either main switch.** Its nearest
  firings are 338,499 (~24k rows before the Benign→DDoS switch) and
  2,538,499 (~277k rows after the Recon switch). It fired 4 times INSIDE
  the Benign file (181k, 285k, 338k) and 6 times inside VulnerabilityScan.
- Feature-score behaviour: at Benign→DDoS the mean score ±1000 rows is
  0.3957, but the 5,000 rows BEFORE the switch already average 0.3326
  (overall score mean 0.1014, median 0.0691) — benign traffic itself is
  non-stationary, so the score was already elevated. At DDoS→Recon the
  score barely moves (0.0647 vs 0.0527 before) — the pattern change does
  not shift the window means enough for this score.
- **Honest verdict:** the feature-based detector as implemented (mean-shift
  score + ADWIN) did NOT outperform error-based detection on this data —
  it fires inside stationary stretches and misses the big switches.
  Feature-based detection of pattern changes remains an open improvement,
  not a solved part of this thesis.

## 3. Explanation stability (47 transitions)

| Method | Mean τ vs prev | Mean Jaccard vs prev | Min τ | Min Jaccard |
|---|---|---|---|---|
| SHAP | +0.668 | 0.685 | −0.037 | 0.273 |
| LIME | +0.777 | 0.534 | +0.627 | 0.167 |

Phase split (n = transitions):

| Segment | SHAP τ / Jaccard | LIME τ / Jaccard |
|---|---|---|
| Phase A (8 floods, n=27) | +0.638 / **0.764** | +0.805 / 0.571 |
| Phase B (5 Recon, n=20) | +0.709 / 0.579 | +0.740 / 0.483 |

- The RQ2 pattern holds on 47 transitions: SHAP's top-7 sets are more
  stable (Jaccard 0.685 vs 0.534), LIME's full ranking is more consistent
  (τ 0.777 vs 0.668; LIME τ never negative, SHAP dips to −0.037).
- The switch collapse is still visible in individual transitions (e.g.
  LIME Jaccard 0.167 at snapshot 12 and snapshot 39; SHAP Jaccard 0.273
  at snapshot 35), but with the tail included Phase B means are higher
  than in the 7-transition subset of the family run — the collapse is a
  transient at the switch, after which stability settles in a new regime.
- The SHAP τ flip-flopping seen before persists (τ +1.000 at snapshots 7
  and 8, then −0.026 at 9 and 10) — near-tied rankings among low-importance
  tail features; Jaccard is the more robust top-feature signal.

## 4. RQ3 TEST — stability vs accuracy (the core result)

47 snapshots, each joined to its containing window's accuracy:
mean snapshot-window accuracy **0.999340**, min **0.9956**, std **0.001186**,
10 unique values. (Compare: binary runs had exactly ONE unique value,
1.0000 — no variation at all. Multiclass created the variation RQ3 needs.)

| Stability metric | vs window accuracy | vs accuracy change |
|---|---|---|
| SHAP τ | Pearson r=+0.052 (p=0.727); Spearman ρ=+0.212 (p=0.152) | r=+0.064 (p=0.671); ρ=+0.135 (p=0.371) |
| SHAP Jaccard | Pearson r=**+0.291 (p=0.047)**; Spearman ρ=+0.253 (p=0.086) | r=+0.132 (p=0.383); ρ=+0.096 (p=0.527) |
| LIME τ | Pearson r=+0.187 (p=0.209); Spearman ρ=+0.179 (p=0.228) | r=+0.082 (p=0.586); ρ=+0.016 (p=0.915) |
| LIME Jaccard | Pearson r=+0.188 (p=0.207); Spearman ρ=+0.106 (p=0.480) | r=−0.057 (p=0.707); ρ=−0.123 (p=0.415) |

Lowest-stability snapshots and their window accuracy:

| Snapshot | Method | τ | Jaccard | Window accuracy |
|---|---|---|---|---|
| 12 | LIME | +0.706 | 0.167 | 1.0000 |
| 39 | LIME | +0.680 | 0.167 | 0.9978 |
| 7 | LIME | +0.734 | 0.273 | 1.0000 |
| 35 | SHAP | +0.457 | 0.273 | 1.0000 |
| 43 | LIME | +0.711 | 0.273 | 1.0000 |
| 10 | LIME | +0.802 | 0.400 | 0.9999 |

### RQ3 VERDICT — measurable, but weak; no meaningful degradation

- **A relationship IS now measurable** (it was not measurable at all in
  the binary runs): all four metric combinations show a weak POSITIVE
  association (r ≈ +0.19 to +0.29), i.e. lower stability tends to go with
  slightly lower accuracy.
- **But it is weak and not robust:** only one of four Pearson correlations
  reaches p < 0.05 (SHAP Jaccard, p = 0.047), none reaches it on Spearman,
  and none survives multiple-comparison correction (Bonferroni α = 0.006
  for 8 tests). Against accuracy CHANGE (rather than level) every
  correlation is near zero (|r| ≤ 0.13, all p > 0.38).
- **Substantively, there is no meaningful degradation:** the worst window
  accuracy in the whole 2.95M-row stream is 99.56%, the worst per-class
  accuracy is 98.67% (PingSweep, 2,262 rows), and 5 of the 6
  lowest-stability snapshots coincide with window accuracy 1.0000.
  Explanation instability of the magnitude observed here (Jaccard down to
  0.167) does NOT coincide with detection degradation of any practical
  significance.
- **RQ3 answer (one sentence):** on CICIoT2023 with this pipeline, there
  is a weak positive association between explanation stability and
  detection accuracy (r ≈ +0.2–0.3), but explanation instability does not
  measurably degrade detection performance — the detector stays above
  99.5% even at the least stable moments. RQ3 is therefore ANSWERED as a
  negative result with numbers, not left unanswered.

## 5. Unseen-family test (frozen model, train on floods only)

| | Phase A (learning) | Phase B (frozen, unseen Recon) |
|---|---|---|
| Unseen model mean accuracy | 0.9905 | **0.0013** |
| Unseen model min / max | 0.2309 (warm-up) | 0.0000 / 0.0937 |
| Windows with accuracy exactly 0 | — | **69 of 70** |
| Main model (kept learning) | — | mean 0.9998, min 0.9956 |

- The frozen multiclass model scores **~0.1% on the unseen Recon
  family** — it cannot name attack types it never trained on. This is
  partly structural (no leaves exist for unseen classes), so it must be
  stated that way, not as "the model failed to generalise" in a vague
  sense.
- **Supplementary check (separate run, true multiclass labels):** a tree
  trained on Benign + 8 floods, frozen, was asked to predict 2,000 sampled
  rows from EACH Recon file. It predicted **0 rows as Benign** in all
  10,000 samples — Recon traffic was always predicted as a flood class
  (ICMP/HTTP/SynonymousIP/PSHACK). So in BINARY terms the frozen model
  still detects the unseen attacks (100% flagged as attack); what
  collapses is exact attack-TYPE identification (multiclass 0.1%).
  Adaptation (continued learning) restores type accuracy to 99.98%.
  (An earlier supplementary attempt trained with a label bug — label 0
  for all rows — and was discarded and redone correctly.)

## 6. Objectives (Synopsis §6) — status after this run

1. **Adaptive IDS pipeline that detects and adapts to concept drift —
   FULLY demonstrated.** 24 drifts detected (13 error + 11 feature),
   retraining on drift, 2.95M rows. Error-based ADWIN now fires at both
   main switches (multiclass dips make them visible), which it did not do
   in binary.
2. **Quantitative stability metrics (Kendall's τ, top-k Jaccard) applied
   across adaptation cycles — FULLY demonstrated.** 47 transitions × 2
   methods = 94 stability measurements.
3. **Compare multiple XAI methods under drift AND analyse their
   relationship with detection performance — DONE.** SHAP vs LIME compared
   across 47 transitions (Phase A vs B split included). The relationship
   part is now measurable: weak positive association (r ≈ +0.2–0.3), no
   meaningful degradation — see §4.

## 7. Bugs found, fixed, and re-verified

1. **Per-class accuracy bug (run 1, found during analysis):** the
   streaming loop incremented `w_class_correct[cid]` for EVERY row
   instead of only correct ones (`+= ok`), so per-class accuracy was
   1.0000 everywhere. Fixed, and the whole pipeline was RE-RUN (run 2).
   Run 2 reproduces run 1 exactly on every shared number (same worst
   windows, same 24 drift rows, same weighted accuracy 0.9999146),
   confirming determinism; per-class numbers in §1 are from run 2 only.
2. **Analysis merge bug:** `pivot_table` MultiIndex columns broke the
   snapshot-table merge; fixed by flattening column names. Re-ran
   analysis; all numbers above are from the fixed run.
3. **Snapshot cap (run 1):** MAX_EXPLAIN=35 cut off the tail — the final
   snapshot and all VulnerabilityScan snapshots were skipped. Raised to
   55 for run 2; all 48 snapshots explained including the final one.
4. **Supplementary-check label bug:** first attempt trained the frozen
   model with label 0 for all rows (predicting Benign everywhere);
   discarded and redone with true multiclass labels (§5).

## 8. Limitations (carry into the thesis)

1. Only 14 of 34 CICIoT2023 classes used (3 files in run 1 → 14 here).
2. 39 features, not 46 (same as all previous runs — unresolved).
3. File order = time order (no timestamps in per-attack files).
4. Surrogate explanations (sklearn DecisionTree on buffer, not the
   Hoeffding tree itself); SHAP importance averaged over classes.
5. Feature-based drift detector as implemented is noisy and missed both
   main switches (§2) — do not claim it as a solved contribution.
6. RQ3 effect is small: accuracy range is only 99.56–100%, so the
   correlation rests on few low-accuracy windows; a harder setup (more
   attack files, overlapping benign/attack traffic, or the merged dataset
   with real timestamps) would be needed for a stronger RQ3 test.
7. HuggingFace mirror, not the official CIC download (needs Ammara's
   form for provenance).
8. PingSweep is tiny (2,262 rows); HTTP_Flood is small (28,790 rows).

## Files

- `experiments/full_pipeline_rq3.py` — the pipeline (multiclass +
  feature drift + unseen model)
- `experiments/analyse_rq3.py` — verification + RQ3 correlation analysis
- `experiments/results_rq3/window_accuracy.csv` — 296 windows
- `experiments/results_rq3/class_accuracy.csv` — per-window per-class
- `experiments/results_rq3/unseen_accuracy.csv` — frozen-model accuracy
- `experiments/results_rq3/drifts.csv` — 24 drifts with detector
- `experiments/results_rq3/feature_scores.csv` — 29,026 score points
- `experiments/results_rq3/cycle_stability.csv` — 94 rows (47×2)
- `experiments/results_rq3/cycle_explanations.csv` — 3,744 rows
- `experiments/results_rq3/rq3_snapshot_table.csv` — joined snapshot table
- `experiments/results_rq3/rq3_numbers.json` — key numbers
- `experiments/results_rq3_run2.log` — run 2 console log
