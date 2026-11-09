"""
Baseline pipeline — Ammara's MS Thesis (CICIoT2023)
====================================================
What this script does (step by step):
  1. Loads a SMALL sample of CICIoT2023 data (3 files: Benign, DDoS-TCP_Flood,
     Recon-PortScan). The files have NO label column — the label comes from
     the file name, so we add it ourselves: Benign = 0, Attack = 1.
  2. Streams the rows one by one (in order: Benign -> DDoS -> Recon) into a
     Hoeffding Tree classifier (river library). This is "online learning":
     the model predicts each flow, then learns from the true label.
  3. Feeds every prediction error (0 = correct, 1 = wrong) into ADWIN,
     a drift detector. When the error rate changes, ADWIN flags a drift.
  4. Prints accuracy and how many drifts ADWIN found (and where).

This is a BASELINE only — a readiness check, not the full experiment.
Full experiments will use more data, time windows, retraining on drift,
and TreeSHAP + LIME explanation stability (Kendall's tau, top-k Jaccard).

Run it from the experiments/ folder:
    ./venv/bin/python baseline_pipeline.py
"""

import os
import time

import pandas as pd
from river import drift, metrics, tree

# ---------------------------------------------------------------------------
# 1. Settings — change these numbers to use more / less data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_PER_FILE = 10_000          # rows to take from EACH file (sample, not full)
RANDOM_STATE = 42            # fixed seed so results are repeatable

# (file name, label) — 0 = Benign, 1 = Attack.
# The stream order matters: Benign first, then attacks, so ADWIN can see
# the distribution change (this is our simple stand-in for concept drift).
FILES = [
    ("BenignTraffic.pcap.csv", 0),
    ("DDoS-TCP_Flood.pcap.csv", 1),
    ("Recon-PortScan.pcap.csv", 1),
]


def load_sample():
    """Load N_PER_FILE rows from each CSV and stack them into one stream.

    Returns a pandas DataFrame with all feature columns + a 'label' column.
    """
    parts = []
    for fname, label in FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"Loading {fname} (first {N_PER_FILE:,} rows) ...")
        df = pd.read_csv(path, nrows=N_PER_FILE)
        df["label"] = label
        parts.append(df)
        print(f"   -> {len(df):,} rows, {len(df.columns) - 1} features, "
              f"label = {label}")

    stream = pd.concat(parts, ignore_index=True)
    print(f"\nTotal stream: {len(stream):,} rows "
          f"({(stream['label'] == 0).sum():,} benign, "
          f"{(stream['label'] == 1).sum():,} attack)\n")
    return stream


def main():
    t0 = time.time()

    # --- Load the sample stream -------------------------------------------
    stream = load_sample()
    feature_cols = [c for c in stream.columns if c != "label"]

    # --- Build the model + drift detector ---------------------------------
    # Hoeffding Tree: a decision tree that learns from a stream, one row
    # at a time. grace_period = how many rows it waits before trying to
    # split a leaf; delta = confidence for splitting.
    model = tree.HoeffdingTreeClassifier(
        grace_period=200, delta=1e-5, leaf_prediction="nb"
    )
    # ADWIN: watches the error stream (0/1). When the average error changes
    # enough, it reports a drift.
    adwin = drift.ADWIN(delta=0.002)
    accuracy = metrics.Accuracy()

    drift_points = []   # row numbers where ADWIN flagged a drift

    # Build plain dicts first: itertuples() cannot handle column names with
    # spaces (e.g. "Protocol Type"), but dicts keep the original names.
    records = stream[feature_cols].to_dict("records")
    labels = stream["label"].tolist()

    # --- Stream: predict, score, learn, check for drift -------------------
    # This is "prequential" evaluation: for each row, FIRST predict,
    # THEN learn from the true label. This is how online IDS is tested.
    for i, (x, y) in enumerate(zip(records, labels)):
        y = int(y)

        y_pred = model.predict_one(x)          # 1. predict
        accuracy.update(y, y_pred)             # 2. score the prediction
        model.learn_one(x, y)                  # 3. learn from the truth

        # 4. feed the error (0 = right, 1 = wrong) to the drift detector
        error = 0 if y_pred == y else 1
        adwin.update(error)
        if adwin.drift_detected:
            drift_points.append(i)

    # --- Report ------------------------------------------------------------
    print("=" * 60)
    print("BASELINE RESULT")
    print("=" * 60)
    print(f"Rows processed      : {len(stream):,}")
    print(f"Accuracy            : {accuracy.get():.4f}")
    print(f"Drifts detected     : {len(drift_points)}")
    if drift_points:
        print(f"Drift positions     : {drift_points}")
        print("  (Benign block ends at row 9,999; DDoS block ends at 19,999)")
    print(f"Time taken          : {time.time() - t0:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
