# Thesis Experiments — Setup Notes (2026-11-06)

## Environment (verified working)
- Python **3.12.3** (virtualenv: `experiments/venv/`)
- numpy 2.5.4 · pandas 3.1.1 · scikit-learn 1.9.1 · river 0.26.1 · shap 0.54.1 · lime (LimeTabularExplainer OK; `lime` has no `__version__` attribute — that is normal)
- Run scripts with: `./venv/bin/python <script>.py` (from `experiments/`)

## Dataset — CICIoT2023
- Official source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
  (Neto et al., CIC, University of New Brunswick — 33 attacks, 7 categories, 105 IoT devices)
- **Official download needs a form** (name/email/org) at cicresearch.ca, so for this
  setup we used the public HuggingFace mirror `bencorn/CIC-IoT-2023` (same original
  per-attack CSV files, 15,941 downloads). For the final thesis, download from the
  official CIC page as well and keep the form confirmation as provenance.
- Local files (`experiments/data/`, 141 MB total):
  - `BenignTraffic.pcap.csv` — 75.1 MB, 362,361 rows
  - `DDoS-TCP_Flood.pcap.csv` — 49.6 MB, 265,776 rows
  - `Recon-PortScan.pcap.csv` — 16.4 MB, 82,284 rows
  - `README_CSV.pdf` — CIC's own CSV readme
  - Total local rows: **710,421**
- **Verified facts about these files:**
  - **39 numeric feature columns, 0 missing values** in checked samples.
  - **NO label column** — the label comes from the file/folder name (one attack per file).
    The baseline adds it: Benign = 0, Attack = 1.
  - ⚠️ Literature says "46 features + label", but these per-attack CSVs have **39 columns**.
    The missing ones (flow_duration, Duration, Srate, Drate, Magnitue, Radius,
    Covariance, Weight, …) are not in these files. Check the full merged CSV before
    writing the methodology section — the feature count must match the actual data.
- Full mirror is ~8.9 GB / 310 files — NOT downloaded. Only the 3 files above are local.

## Baseline run (verified 2026-11-06)
- Script: `experiments/baseline_pipeline.py`
- Sample: 10,000 rows per file → 30,000-row stream (Benign → DDoS → Recon)
- Model: river `HoeffdingTreeClassifier` (grace_period=200, delta=1e-5, leaf_prediction="nb")
- Drift: river `ADWIN` (delta=0.002) on the prediction-error stream
- **Result: accuracy 0.9933, 2 drifts detected at rows 20,031 and 20,223**
  (right after the DDoS → Recon switch at row 20,000 — ADWIN caught the change).
  No drift was flagged at the Benign → DDoS switch (row 10,000); the error rate
  there did not shift enough for ADWIN's threshold.
- Runtime: ~2.3 seconds for 30k rows.

## Still needed before full experiments
1. **Full data**: download the remaining attack files (or the merged CSV) from the
   official CIC page; decide binary vs. multiclass (34 classes) labels.
2. **Time windows**: the per-attack files have no timestamp column, so real
   time-ordered windows need the merged dataset or file-order assumptions —
   document the choice in the methodology.
3. **Retrain-on-drift**: baseline only detects drift; full pipeline must reset/
   retrain the tree when ADWIN fires.
4. **Explanation stability**: TreeSHAP + LIME on the tree, Kendall's tau +
   top-k Jaccard across adaptation cycles (the thesis core — not built yet).
5. **Feature count**: resolve the 39-vs-46 feature question before writing Chapter 3.
6. **Hardware**: this VM has 4 CPUs / 7.7 GB RAM (only ~600 MB free during setup).
   Full-data runs should use chunked/streaming loads, never load all 8.9 GB at once.
