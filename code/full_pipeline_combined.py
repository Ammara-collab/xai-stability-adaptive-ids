"""
Full experiment pipeline — Ammara's MS Thesis (CICIoT2023)
===========================================================
What this script does:
  1. Streams 7 local CICIoT2023 files in order (Benign -> TCP -> UDP -> SYN
     -> ICMP -> HTTP -> PSHACK, all DDoS flood family) as time windows of
     10,000 rows. This file order is
     our stand-in for time order, because the per-attack CSVs have NO
     timestamp column (documented assumption — see notes at the bottom).
  2. Online learning with a Hoeffding Tree (river) + prequential evaluation
     (predict, then learn). Accuracy is tracked per window.
  3. ADWIN watches the prediction-error stream. WHEN ADWIN fires, the tree
     is RESET (a fresh HoeffdingTreeClassifier) and learning resumes on the
     post-drift stream. That is what "retrain on drift" means here: discard
     the old tree and start a new one on the new distribution.
  4. EXPLANATION STABILITY (the thesis core): explanations are computed on
     a FIXED sample of 30 instances (same 30 instances every time, so
     snapshots are comparable), at two kinds of moments:
       - DRIFT snapshots: when ADWIN fires, just BEFORE the reset — this is
         the end of one adaptation cycle.
       - TIME CHECKPOINTS: every 100,000 rows, without resetting — the model
         updates continuously (online learning), so checkpoints track how
         explanations move as the model keeps learning.
       - One FINAL snapshot at the end of the stream.
     Each snapshot fits a surrogate sklearn DecisionTree to the current
     buffer and explains it with TreeSHAP and LIME (river's Hoeffding tree
     has no native SHAP/LIME support, so a surrogate tree trained on the
     same window data is used — a standard, documented choice; see notes).
     Stability between consecutive snapshots is measured with Kendall's tau
     (ranking agreement of mean |importance| vectors) and top-7 Jaccard
     (overlap of the top-7 feature sets). SHAP and LIME are reported
     separately (RQ2).
  5. RQ3: per-window accuracy is recorded alongside the stability numbers,
     so we can see whether accuracy drops when explanations are unstable.

Outputs (experiments/results/):
  - window_accuracy.csv, drifts.csv, cycle_stability.csv,
    cycle_explanations.csv, RESULTS_SUMMARY.md

Run it from the experiments/ folder:
    ./venv/bin/python full_pipeline.py
"""

import os
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.tree import DecisionTreeClassifier
from river import drift, metrics, tree
import shap
from lime.lime_tabular import LimeTabularExplainer

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results_family", "combined")
os.makedirs(RESULTS_DIR, exist_ok=True)

# (file name, label) — 0 = Benign, 1 = Attack. Order = time order.
# Phase A (DDoS flood family, 8 variants) then Phase B (Recon family, 5).
FILES = [
    ("BenignTraffic.pcap.csv", 0),
    ("DDoS-TCP_Flood.pcap.csv", 1),
    ("DDoS-UDP_Flood.pcap.csv", 1),
    ("DDoS-SYN_Flood.pcap.csv", 1),
    ("DDoS-ICMP_Flood.pcap.csv", 1),
    ("DDoS-HTTP_Flood.pcap.csv", 1),
    ("DDoS-PSHACK_Flood.pcap.csv", 1),
    ("DDoS-RSTFINFlood.pcap.csv", 1),
    ("DDoS-SynonymousIP_Flood.pcap.csv", 1),
    ("Recon-PortScan.pcap.csv", 1),
    ("Recon-PingSweep.pcap.csv", 1),
    ("Recon-OSScan.pcap.csv", 1),
    ("Recon-HostDiscovery.pcap.csv", 1),
    ("VulnerabilityScan.pcap.csv", 1),
]

WINDOW_SIZE = 10_000        # rows per time window (accuracy per window)
BUFFER_PER_CLASS = 5_000    # rolling buffer: last 5,000 rows of EACH class
                            # (keeps both classes present for the surrogate,
                            # even late in the stream when only attacks flow)
CHUNK_SIZE = 10_000         # pandas read chunk (streaming, low memory)
CHECKPOINT_EVERY = 100_000  # rows between time checkpoints (explanations)
MAX_EXPLAIN = 35            # safety cap on explanation snapshots
MIN_BUFFER_FOR_EXPLAIN = 200  # need at least this many rows to fit a surrogate
LIME_NUM_SAMPLES = 500      # LIME perturbations per instance (speed vs quality)
TOP_K = 7                   # top-k for Jaccard (matches Alqahtani's top-7)

# Fixed row positions taken from EACH file for the explanation sample.
# Same 30 instances are explained at every cycle, so cycles are comparable.
EXPLAIN_POSITIONS = [1000, 10000, 30000, 50000, 70000]


def make_model():
    """A fresh Hoeffding Tree (same settings as the baseline)."""
    return tree.HoeffdingTreeClassifier(
        grace_period=200, delta=1e-5, leaf_prediction="nb"
    )


def make_adwin():
    """A fresh ADWIN drift detector."""
    return drift.ADWIN(delta=0.002)


# ---------------------------------------------------------------------------
# 2. Fixed explanation sample (pre-pass: read only the first 80,001 rows
#    of each file and pick the fixed positions).
# ---------------------------------------------------------------------------
def collect_explain_sample():
    """Return (X_sample array, feature_cols, sample labels, sample sources).

    10 fixed rows per file -> 30 instances total. Deterministic, no random
    sampling, so the exact same instances are explained at every cycle.
    """
    parts, labels, sources = [], [], []
    feature_cols = None
    for fname, label in FILES:
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, nrows=EXPLAIN_POSITIONS[-1] + 1)
        cols = [c for c in df.columns]
        if feature_cols is None:
            feature_cols = cols
        wanted = [p for p in EXPLAIN_POSITIONS if p < len(df)]
        parts.append(df.iloc[wanted])
        labels.extend([label] * len(wanted))
        sources.extend([fname] * len(wanted))
        print(f"  explain sample: {fname}: {len(wanted)} rows")
    X_sample = pd.concat(parts, ignore_index=True)[feature_cols].to_numpy(
        dtype=float
    )
    print(f"  explain sample total: {X_sample.shape[0]} instances, "
          f"{X_sample.shape[1]} features\n")
    return X_sample, feature_cols, labels, sources


# ---------------------------------------------------------------------------
# 3. Explanation helpers
# ---------------------------------------------------------------------------
def shap_ranking(surrogate, X_sample):
    """Mean |SHAP value| per feature for the surrogate tree.

    Returns a 1-D array (one value per feature). Handles the different
    shapes shap returns across versions (list / 2-D / 3-D).
    """
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        # list per class -> take the Attack class (index 1)
        arr = sv[1] if len(sv) > 1 else sv[0]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        arr = sv[:, :, 1]  # (samples, features, classes) -> Attack class
    else:
        arr = np.asarray(sv)
    return np.abs(arr).mean(axis=0)


def lime_ranking(surrogate, X_buffer, X_sample, feature_cols):
    """Mean |LIME weight| per feature, averaged over the sample instances.

    LIME explains the surrogate (sklearn predict_proba), because river's
    Hoeffding tree has no predict_proba for arrays.
    """
    feature_index = {name: i for i, name in enumerate(feature_cols)}
    explainer = LimeTabularExplainer(
        X_buffer,
        feature_names=feature_cols,
        class_names=["Benign", "Attack"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )
    weights = np.zeros((len(X_sample), len(feature_cols)))
    for i, x in enumerate(X_sample):
        exp = explainer.explain_instance(
            x, surrogate.predict_proba,
            num_features=10, num_samples=LIME_NUM_SAMPLES,
        )
        for feat_name, w in exp.as_list():
            j = feature_index.get(feat_name)
            if j is not None:
                weights[i, j] = w
    return np.abs(weights).mean(axis=0)


def top_k_names(mean_importance, feature_cols, k=TOP_K):
    """Names of the k most important features, best first."""
    order = np.argsort(-mean_importance)
    return [feature_cols[i] for i in order[:k]]


def jaccard(set_a, set_b):
    """Overlap of two feature sets: |A n B| / |A u B| (0..1)."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# 4. Explain one cycle: fit a surrogate on the buffer, explain the fixed
#    sample with SHAP and LIME, return the rankings.
# ---------------------------------------------------------------------------
def explain_cycle(buffer_X, X_sample, feature_cols):
    """Fit surrogate on buffer, return dict with SHAP/LIME rankings."""
    surrogate = DecisionTreeClassifier(random_state=42)
    surrogate.fit(buffer_X[0], buffer_X[1])

    shap_imp = shap_ranking(surrogate, X_sample)
    lime_imp = lime_ranking(surrogate, buffer_X[0], X_sample, feature_cols)
    return {
        "shap_imp": shap_imp,
        "lime_imp": lime_imp,
        "shap_top7": top_k_names(shap_imp, feature_cols),
        "lime_top7": top_k_names(lime_imp, feature_cols),
    }


# ---------------------------------------------------------------------------
# 5. Main stream
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 64)
    print("COMBINED PIPELINE — CICIoT2023 (Benign -> 8x DDoS Flood -> 5x Recon)")
    print("=" * 64)

    # --- Fixed explanation sample (pre-pass) ------------------------------
    print("Collecting fixed explanation sample ...")
    X_sample, feature_cols, _, _ = collect_explain_sample()

    # --- Model, drift detector, metrics -----------------------------------
    model = make_model()
    adwin = make_adwin()
    accuracy = metrics.Accuracy()

    # Rolling per-class buffers of recent rows for the surrogate tree:
    # the last 5,000 benign rows and the last 5,000 attack rows. Keeping
    # both classes matters — late in the stream only attacks flow, and a
    # surrogate trained on one class only cannot produce LIME explanations.
    buffer_benign = deque(maxlen=BUFFER_PER_CLASS)
    buffer_attack = deque(maxlen=BUFFER_PER_CLASS)

    # Per-window bookkeeping.
    window_correct, window_total = 0, 0
    window_benign, window_attack = 0, 0
    window_rows = []          # dicts: window, start_row, end_row, n, accuracy
    drift_rows = []           # global row numbers where ADWIN fired
    snapshots = []            # dicts: snapshot_id, row, kind, shap/lime ranks
    skipped = []              # (snapshot_id, row, kind, reason) when skipped

    n_rows = 0                # global row counter (0-based, +1 after a row)
    cycle_id = 0              # adaptation cycle number (increments on drift)
    snapshot_id = 0
    explain_count = 0

    def take_snapshot(end_row, kind):
        """Explain the current model (surrogate on buffer), record a snapshot.

        kind: 'drift' (just before a reset), 'checkpoint' (every 100k rows),
        or 'final' (end of stream). A snapshot is skipped (and the reason
        recorded) if the buffer is too small, holds only one class, or the
        explanation cap is reached — a failed explanation must never kill
        the whole run.
        """
        nonlocal explain_count, snapshot_id
        sid = snapshot_id
        snapshot_id += 1
        if explain_count >= MAX_EXPLAIN:
            skipped.append((sid, end_row, kind, "cap reached"))
            return False
        buf = list(buffer_benign) + list(buffer_attack)
        if len(buf) < MIN_BUFFER_FOR_EXPLAIN:
            skipped.append((sid, end_row, kind,
                            f"buffer too small ({len(buf)} rows)"))
            return False
        classes = {r[1] for r in buf}
        if len(classes) < 2:
            skipped.append((sid, end_row, kind,
                            "buffer holds a single class only"))
            return False
        try:
            buf_X = np.array([[r[0][c] for c in feature_cols] for r in buf],
                             dtype=float)
            buf_y = np.array([r[1] for r in buf], dtype=int)
            # DATA CLEANING (documented): the 'Rate' feature contains
            # infinite values (rate = packets / zero time interval).
            # river's Hoeffding tree accepts them, but sklearn's
            # DecisionTree (the surrogate) rejects inf, so replace
            # +-inf with the column median of this buffer.
            buf_X = np.where(np.isinf(buf_X), np.nan, buf_X)
            col_median = np.nanmedian(buf_X, axis=0)
            col_median = np.where(np.isnan(col_median), 0.0, col_median)
            nan_idx = np.where(np.isnan(buf_X))
            buf_X[nan_idx] = np.take(col_median, nan_idx[1])
            res = explain_cycle((buf_X, buf_y), X_sample, feature_cols)
        except Exception as e:  # noqa: BLE001 - record and keep streaming
            skipped.append((sid, end_row, kind, f"explainer error: {e}"))
            return False
        res["snapshot_id"] = sid
        res["cycle_id"] = cycle_id
        res["end_row"] = end_row
        res["kind"] = kind
        res["buffer_rows"] = len(buf)
        snapshots.append(res)
        explain_count += 1
        print(f"  [snapshot {sid} | {kind} | cycle {cycle_id}] explained "
              f"at row {end_row:,} (buffer {len(buf):,} rows)")
        print(f"      SHAP top-7: {res['shap_top7']}")
        print(f"      LIME top-7: {res['lime_top7']}")
        return True

    # --- Stream the 3 files in order --------------------------------------
    for fname, label in FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"\nStreaming {fname} (label={label}) ...")
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            chunk = chunk[feature_cols]
            records = chunk.to_dict("records")
            for x in records:
                y = label

                y_pred = model.predict_one(x)       # 1. predict
                ok = int(y_pred == y)
                accuracy.update(y, y_pred)          # 2. score
                model.learn_one(x, y)               # 3. learn
                if y == 0:
                    buffer_benign.append((x, y))
                else:
                    buffer_attack.append((x, y))

                window_correct += ok
                window_total += 1
                if y == 0:
                    window_benign += 1
                else:
                    window_attack += 1
                n_rows += 1

                # 4. window boundary -> record per-window accuracy
                if n_rows % WINDOW_SIZE == 0:
                    w = n_rows // WINDOW_SIZE
                    window_rows.append({
                        "window": w,
                        "start_row": n_rows - WINDOW_SIZE,
                        "end_row": n_rows - 1,
                        "n": window_total,
                        "accuracy": window_correct / window_total,
                        "n_benign": window_benign,
                        "n_attack": window_attack,
                    })
                    window_correct, window_total = 0, 0
                    window_benign, window_attack = 0, 0

                # 5. time checkpoint every 100k rows (no reset — the model
                #    keeps learning; the checkpoint just records how its
                #    explanations look at this point in time).
                if n_rows % CHECKPOINT_EVERY == 0:
                    print(f"\n  --- checkpoint at row {n_rows - 1:,} ---")
                    take_snapshot(n_rows - 1, kind="checkpoint")

                # 6. drift check on the error stream
                adwin.update(1 - ok)
                if adwin.drift_detected:
                    drift_rows.append(n_rows - 1)
                    print(f"\n  *** DRIFT at row {n_rows - 1:,} "
                          f"(cycle {cycle_id} ends, accuracy so far "
                          f"{accuracy.get():.4f}) ***")
                    take_snapshot(n_rows - 1, kind="drift")
                    # RETRAIN ON DRIFT: discard the old tree and ADWIN,
                    # start fresh on the post-drift stream.
                    model = make_model()
                    adwin = make_adwin()
                    cycle_id += 1

            print(f"  ... {n_rows:,} rows streamed, "
                  f"accuracy {accuracy.get():.4f}, "
                  f"drifts {len(drift_rows)}, cycles explained {explain_count}")

    # --- Last partial window -------------------------------------------------
    if window_total > 0:
        w = n_rows // WINDOW_SIZE + 1
        window_rows.append({
            "window": w,
            "start_row": n_rows - window_total,
            "end_row": n_rows - 1,
            "n": window_total,
            "accuracy": window_correct / window_total,
            "n_benign": window_benign,
            "n_attack": window_attack,
        })

    # --- Save window + drift results NOW (before the final explanation,
    #     so they survive even if explaining the last model fails) ---------
    pd.DataFrame(window_rows).to_csv(
        os.path.join(RESULTS_DIR, "window_accuracy.csv"), index=False)
    pd.DataFrame({"drift_index": range(len(drift_rows)),
                  "row": drift_rows}).to_csv(
        os.path.join(RESULTS_DIR, "drifts.csv"), index=False)

    # --- Final snapshot: explain the last model at end of stream -----------
    print("\nEnd of stream — explaining final model ...")
    take_snapshot(n_rows - 1, kind="final")

    # -----------------------------------------------------------------------
    # 6. Stability across consecutive snapshots (RQ1 + RQ2)
    # -----------------------------------------------------------------------
    stability_rows = []
    for k in range(1, len(snapshots)):
        prev, cur = snapshots[k - 1], snapshots[k]
        for method in ("shap", "lime"):
            tau_prev, _ = kendalltau(prev[f"{method}_imp"],
                                     cur[f"{method}_imp"])
            jac_prev = jaccard(prev[f"{method}_top7"], cur[f"{method}_top7"])
            tau_0, _ = kendalltau(snapshots[0][f"{method}_imp"],
                                  cur[f"{method}_imp"])
            jac_0 = jaccard(snapshots[0][f"{method}_top7"],
                            cur[f"{method}_top7"])
            stability_rows.append({
                "snapshot": cur["snapshot_id"],
                "cycle": cur["cycle_id"],
                "kind": cur["kind"],
                "end_row": cur["end_row"],
                "prev_kind": prev["kind"],
                "method": method.upper(),
                "kendall_tau_vs_prev": round(float(tau_prev), 4),
                "jaccard_top7_vs_prev": round(float(jac_prev), 4),
                "kendall_tau_vs_first": round(float(tau_0), 4),
                "jaccard_top7_vs_first": round(float(jac_0), 4),
            })

    # -----------------------------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------------------------
    pd.DataFrame(stability_rows).to_csv(
        os.path.join(RESULTS_DIR, "cycle_stability.csv"), index=False)

    expl_rows = []
    for c in snapshots:
        for method in ("shap", "lime"):
            imp = c[f"{method}_imp"]
            order = np.argsort(-imp)
            for rank, i in enumerate(order, start=1):
                expl_rows.append({
                    "snapshot": c["snapshot_id"],
                    "cycle": c["cycle_id"],
                    "kind": c["kind"],
                    "end_row": c["end_row"],
                    "method": method.upper(),
                    "rank": rank,
                    "feature": feature_cols[i],
                    "mean_abs_importance": round(float(imp[i]), 6),
                })
    pd.DataFrame(expl_rows).to_csv(
        os.path.join(RESULTS_DIR, "cycle_explanations.csv"), index=False)

    # --- Console report ------------------------------------------------------
    print("\n" + "=" * 64)
    print("FULL PIPELINE RESULT")
    print("=" * 64)
    print(f"Rows processed      : {n_rows:,}")
    print(f"Overall accuracy    : {accuracy.get():.4f}")
    print(f"Windows             : {len(window_rows)} "
          f"({WINDOW_SIZE:,} rows each)")
    print(f"Drifts detected     : {len(drift_rows)} at rows {drift_rows}")
    print(f"Snapshots explained : {len(snapshots)} "
          f"(skipped: {len(skipped)})")
    for s in skipped:
        print(f"  skipped snapshot {s[0]} ({s[2]} at row {s[1]:,}): {s[3]}")
    for s in stability_rows:
        print(f"  snapshot {s['snapshot']} [{s['prev_kind']}->{s['kind']}] "
              f"{s['method']}: tau vs prev {s['kendall_tau_vs_prev']:+.3f}, "
              f"Jaccard vs prev {s['jaccard_top7_vs_prev']:.3f}, "
              f"tau vs first {s['kendall_tau_vs_first']:+.3f}, "
              f"Jaccard vs first {s['jaccard_top7_vs_first']:.3f}")
    print(f"Time taken          : {time.time() - t0:.1f} seconds")
    print("=" * 64)

    # Save the numbers needed for the summary markdown.
    summary_data = {
        "n_rows": n_rows,
        "overall_accuracy": accuracy.get(),
        "n_windows": len(window_rows),
        "drift_rows": drift_rows,
        "n_snapshots": len(snapshots),
        "skipped": skipped,
        "stability_rows": stability_rows,
        "window_rows": window_rows,
        "snapshots": snapshots,
        "time_s": time.time() - t0,
    }
    return summary_data


if __name__ == "__main__":
    main()
