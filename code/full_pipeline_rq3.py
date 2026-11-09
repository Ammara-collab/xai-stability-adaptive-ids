"""
RQ3 harder experiment — Ammara's MS Thesis (CICIoT2023)
=========================================================
Why this run exists:
  Runs 1 + family (binary labels) left RQ3 UNANSWERED: accuracy was 1.0000
  at every snapshot, so there was zero variation to correlate with
  explanation instability. This run makes the task harder in three ways:

  1. MULTICLASS labels (14 classes: Benign + 13 attack types, one class per
     file) instead of binary, so a new attack family can actually confuse
     the classifier and create accuracy variation (incl. per-class drops).
  2. FEATURE-BASED drift detection instead of (only) error-based ADWIN:
     every 100 rows a mean-shift score is computed between the last 1,000
     rows and the 1,000 rows before them (per-feature standardised shift,
     averaged over the 39 features) and fed to ADWIN. Error-based ADWIN is
     kept as a monitor for comparison. EITHER detector firing triggers a
     snapshot + reset (the detector is logged), because both are legitimate
     drift signals.
  3. TRAIN-ON-FEWER / TEST-ON-UNSEEN: a second Hoeffding tree (model_unseen)
     learns ONLY on Benign + the 8 DDoS floods (Phase A) and is then FROZEN
     for Phase B (5 Recon attacks) — predict only, no learning. Its Phase B
     accuracy measures generalisation to an unseen attack family.

  Stream: Benign -> 8 DDoS floods -> 5 Recon (file order = time order,
  documented assumption: per-attack CSVs have no timestamp column).
  Windows of 10,000 rows; checkpoints every 100,000 rows; TreeSHAP + LIME
  stability (Kendall's tau + top-7 Jaccard) on a fixed 70-instance sample,
  exactly as in the earlier runs. RQ3 is tested by correlating per-snapshot
  stability with per-window accuracy (done in analyse_rq3.py afterwards).

Outputs (experiments/results_rq3/):
  - window_accuracy.csv (per-window accuracy, main + unseen model, phase)
  - class_accuracy.csv (per-window per-class accuracy)
  - drifts.csv (drift row + which detector fired)
  - feature_scores.csv (the feature drift score stream, every 100 rows)
  - unseen_accuracy.csv (per-window accuracy of the frozen unseen model)
  - cycle_stability.csv, cycle_explanations.csv
  - RESULTS_SUMMARY_RQ3.md (written by analyse_rq3.py)

Run from the experiments/ folder:
    ./venv/bin/python full_pipeline_rq3.py
"""

import os
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.tree import DecisionTreeClassifier
from river import drift, metrics, tree
import shap
from lime.lime_tabular import LimeTabularExplainer

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results_rq3")
os.makedirs(RESULTS_DIR, exist_ok=True)

# (file name, class id, phase, class name). 14 classes total.
# Phase A = DDoS flood family (8 variants); Phase B = Recon family (5).
FILES = [
    ("BenignTraffic.pcap.csv", 0, "benign", "Benign"),
    ("DDoS-TCP_Flood.pcap.csv", 1, "A", "TCP_Flood"),
    ("DDoS-UDP_Flood.pcap.csv", 2, "A", "UDP_Flood"),
    ("DDoS-SYN_Flood.pcap.csv", 3, "A", "SYN_Flood"),
    ("DDoS-ICMP_Flood.pcap.csv", 4, "A", "ICMP_Flood"),
    ("DDoS-HTTP_Flood.pcap.csv", 5, "A", "HTTP_Flood"),
    ("DDoS-PSHACK_Flood.pcap.csv", 6, "A", "PSHACK_Flood"),
    ("DDoS-RSTFINFlood.pcap.csv", 7, "A", "RSTFIN_Flood"),
    ("DDoS-SynonymousIP_Flood.pcap.csv", 8, "A", "SynonymousIP_Flood"),
    ("Recon-PortScan.pcap.csv", 9, "B", "PortScan"),
    ("Recon-PingSweep.pcap.csv", 10, "B", "PingSweep"),
    ("Recon-OSScan.pcap.csv", 11, "B", "OSScan"),
    ("Recon-HostDiscovery.pcap.csv", 12, "B", "HostDiscovery"),
    ("VulnerabilityScan.pcap.csv", 13, "B", "VulnerabilityScan"),
]
CLASS_NAMES = [c[3] for c in FILES]
N_CLASSES = len(FILES)

WINDOW_SIZE = 10_000
BUFFER_PER_CLASS = 1_000     # rolling buffer: last 1,000 rows of EACH class
CHUNK_SIZE = 10_000
CHECKPOINT_EVERY = 100_000
MAX_EXPLAIN = 55
MIN_BUFFER_FOR_EXPLAIN = 200
LIME_NUM_SAMPLES = 500
TOP_K = 7

# Feature-drift detector settings.
FEAT_WINDOW = 1_000          # rows in each of the two windows compared
FEAT_EVERY = 100             # rows between score computations
FEAT_EPS = 1e-9

EXPLAIN_POSITIONS = [1000, 10000, 30000, 50000, 70000]


def make_model():
    return tree.HoeffdingTreeClassifier(
        grace_period=200, delta=1e-5, leaf_prediction="nb"
    )


def make_adwin():
    return drift.ADWIN(delta=0.002)


# ---------------------------------------------------------------------------
# 2. Fixed explanation sample (5 fixed rows per file -> up to 70 instances)
# ---------------------------------------------------------------------------
def collect_explain_sample():
    parts, labels, sources = [], [], []
    feature_cols = None
    for fname, cid, phase, cname in FILES:
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, nrows=EXPLAIN_POSITIONS[-1] + 1)
        cols = [c for c in df.columns]
        if feature_cols is None:
            feature_cols = cols
        wanted = [p for p in EXPLAIN_POSITIONS if p < len(df)]
        parts.append(df.iloc[wanted])
        labels.extend([cid] * len(wanted))
        sources.extend([fname] * len(wanted))
        print(f"  explain sample: {fname}: {len(wanted)} rows")
    X_sample = pd.concat(parts, ignore_index=True)[feature_cols].to_numpy(
        dtype=float
    )
    print(f"  explain sample total: {X_sample.shape[0]} instances, "
          f"{X_sample.shape[1]} features\n")
    return X_sample, feature_cols, labels, sources


# ---------------------------------------------------------------------------
# 3. Explanation helpers (multiclass-aware)
# ---------------------------------------------------------------------------
def shap_ranking(surrogate, X_sample):
    """Mean |SHAP value| per feature, averaged over samples AND classes.

    For a multiclass surrogate, shap returns one array per class (list) or
    a 3-D array (samples, features, classes). Averaging over classes gives
    a single importance vector per feature, comparable across snapshots.
    """
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        arrs = [np.abs(np.asarray(a)) for a in sv]
        stacked = np.stack(arrs, axis=0)          # (classes, samples, feats)
        return stacked.mean(axis=(0, 1))
    arr = np.asarray(sv)
    if arr.ndim == 3:                             # (samples, feats, classes)
        return np.abs(arr).mean(axis=(0, 2))
    return np.abs(arr).mean(axis=0)               # (samples, feats)


def lime_ranking(surrogate, X_buffer, X_sample, feature_cols):
    """Mean |LIME weight| per feature, averaged over the sample instances."""
    feature_index = {name: i for i, name in enumerate(feature_cols)}
    explainer = LimeTabularExplainer(
        X_buffer,
        feature_names=feature_cols,
        class_names=CLASS_NAMES,
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
    order = np.argsort(-mean_importance)
    return [feature_cols[i] for i in order[:k]]


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def explain_cycle(buffer_X, buffer_y, X_sample, feature_cols):
    surrogate = DecisionTreeClassifier(random_state=42)
    surrogate.fit(buffer_X, buffer_y)
    shap_imp = shap_ranking(surrogate, X_sample)
    lime_imp = lime_ranking(surrogate, buffer_X, X_sample, feature_cols)
    return {
        "shap_imp": shap_imp,
        "lime_imp": lime_imp,
        "shap_top7": top_k_names(shap_imp, feature_cols),
        "lime_top7": top_k_names(lime_imp, feature_cols),
    }


# ---------------------------------------------------------------------------
# 4. Main stream
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 64)
    print("RQ3 PIPELINE — MULTICLASS (14 classes) + FEATURE-BASED DRIFT")
    print("Benign -> 8x DDoS Flood -> 5x Recon | unseen model frozen in B")
    print("=" * 64)

    print("Collecting fixed explanation sample ...")
    X_sample, feature_cols, _, _ = collect_explain_sample()

    model = make_model()
    model_unseen = make_model()   # learns only in Benign + Phase A
    adwin_err = make_adwin()      # error-based (monitor + can trigger reset)
    adwin_feat = make_adwin()     # feature-based (main drift detector)
    accuracy = metrics.Accuracy()

    # Rolling per-class buffers for the surrogate tree.
    buffers = {cid: deque(maxlen=BUFFER_PER_CLASS) for _, cid, _, _ in FILES}

    # Feature-drift windows: last 2*FEAT_WINDOW rows as float arrays.
    feat_rows = deque(maxlen=2 * FEAT_WINDOW)

    window_rows = []
    class_rows = []        # per-window per-class accuracy rows
    unseen_rows = []       # per-window accuracy of the frozen unseen model
    drift_rows = []        # (row, detector) tuples
    feat_scores = []       # (row, score) every FEAT_EVERY rows
    snapshots = []
    skipped = []

    # Per-window bookkeeping.
    w_correct, w_total = 0, 0
    w_class_correct = {cid: 0 for _, cid, _, _ in FILES}
    w_class_total = {cid: 0 for _, cid, _, _ in FILES}
    w_unseen_correct, w_unseen_total = 0, 0
    w_phase = "benign"

    n_rows = 0
    cycle_id = 0
    snapshot_id = 0
    explain_count = 0

    def clean_row(x):
        """Row dict -> float array (inf/NaN -> 0) for the feature statistic."""
        arr = np.array([x[c] for c in feature_cols], dtype=float)
        return np.where(np.isfinite(arr), arr, 0.0)

    def feature_drift_score():
        """Mean standardised shift between the last two FEAT_WINDOW blocks.

        Returns None until 2*FEAT_WINDOW rows are buffered.
        """
        if len(feat_rows) < 2 * FEAT_WINDOW:
            return None
        arr = np.array(feat_rows, dtype=float)
        ref, cur = arr[:FEAT_WINDOW], arr[FEAT_WINDOW:]
        pooled_std = np.std(arr, axis=0)
        shift = np.abs(cur.mean(axis=0) - ref.mean(axis=0))
        z = shift / (pooled_std + FEAT_EPS)
        return float(np.mean(z))

    def take_snapshot(end_row, kind):
        nonlocal explain_count, snapshot_id
        sid = snapshot_id
        snapshot_id += 1
        if explain_count >= MAX_EXPLAIN:
            skipped.append((sid, end_row, kind, "cap reached"))
            return False
        buf = []
        for cid in sorted(buffers):
            buf.extend(buffers[cid])
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
            # DATA CLEANING (documented): 'Rate' contains inf values.
            # Replace +-inf with the column median of this buffer.
            buf_X = np.where(np.isinf(buf_X), np.nan, buf_X)
            col_median = np.nanmedian(buf_X, axis=0)
            col_median = np.where(np.isnan(col_median), 0.0, col_median)
            nan_idx = np.where(np.isnan(buf_X))
            buf_X[nan_idx] = np.take(col_median, nan_idx[1])
            res = explain_cycle(buf_X, buf_y, X_sample, feature_cols)
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

    def close_window(end_row, phase):
        nonlocal w_correct, w_total, w_unseen_correct, w_unseen_total
        nonlocal w_class_correct, w_class_total
        w = end_row // WINDOW_SIZE + 1
        window_rows.append({
            "window": w,
            "start_row": end_row - w_total + 1 if w_total else end_row,
            "end_row": end_row,
            "n": w_total,
            "accuracy": w_correct / w_total if w_total else float("nan"),
            "acc_unseen": (w_unseen_correct / w_unseen_total
                           if w_unseen_total else float("nan")),
            "n_benign": w_class_total.get(0, 0),
            "n_attack": w_total - w_class_total.get(0, 0),
            "phase": phase,
        })
        for cid in sorted(w_class_total):
            n_c = w_class_total[cid]
            if n_c > 0:
                class_rows.append({
                    "window": w,
                    "end_row": end_row,
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "n": n_c,
                    "correct": w_class_correct[cid],
                    "accuracy": w_class_correct[cid] / n_c,
                })
        unseen_rows.append({
            "window": w,
            "end_row": end_row,
            "phase": phase,
            "n": w_unseen_total,
            "accuracy": (w_unseen_correct / w_unseen_total
                         if w_unseen_total else float("nan")),
        })
        w_correct, w_total = 0, 0
        w_unseen_correct, w_unseen_total = 0, 0
        w_class_correct = {cid: 0 for _, cid, _, _ in FILES}
        w_class_total = {cid: 0 for _, cid, _, _ in FILES}

    def reset_on_drift(end_row, detector):
        """Snapshot, then discard model + detectors and start a new cycle."""
        nonlocal model, adwin_err, adwin_feat, cycle_id
        drift_rows.append((end_row, detector))
        print(f"\n  *** DRIFT ({detector}) at row {end_row:,} "
              f"(cycle {cycle_id} ends, accuracy so far "
              f"{accuracy.get():.4f}) ***")
        take_snapshot(end_row, kind="drift")
        model = make_model()
        adwin_err = make_adwin()
        adwin_feat = make_adwin()
        feat_rows.clear()
        cycle_id += 1

    # --- Stream the files in order ---------------------------------------
    for fname, cid, phase, cname in FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"\nStreaming {fname} (class={cid} {cname}, phase={phase}) ...")
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            chunk = chunk[feature_cols]
            records = chunk.to_dict("records")
            for x in records:
                y = cid

                # --- main model: predict -> score -> learn (prequential) --
                y_pred = model.predict_one(x)
                ok = int(y_pred == y)
                accuracy.update(y, y_pred)
                model.learn_one(x, y)
                buffers[cid].append((x, y))

                # --- unseen model: learns only in Benign + Phase A --------
                if phase in ("benign", "A"):
                    y_pred_u = model_unseen.predict_one(x)
                    ok_u = int(y_pred_u == y)
                    model_unseen.learn_one(x, y)
                else:  # Phase B: frozen, predict only
                    y_pred_u = model_unseen.predict_one(x)
                    ok_u = int(y_pred_u == y)
                w_unseen_correct += ok_u
                w_unseen_total += 1

                # --- feature-drift statistic ------------------------------
                feat_rows.append(clean_row(x))

                # --- window bookkeeping ------------------------------------
                w_correct += ok
                w_total += 1
                w_class_correct[cid] += ok
                w_class_total[cid] += 1
                w_phase = phase
                n_rows += 1

                if n_rows % WINDOW_SIZE == 0:
                    close_window(n_rows - 1, w_phase)

                if n_rows % CHECKPOINT_EVERY == 0:
                    print(f"\n  --- checkpoint at row {n_rows - 1:,} ---")
                    take_snapshot(n_rows - 1, kind="checkpoint")

                # --- feature-based drift check (every FEAT_EVERY rows) -----
                if n_rows % FEAT_EVERY == 0:
                    score = feature_drift_score()
                    if score is not None:
                        feat_scores.append((n_rows - 1, score))
                        adwin_feat.update(score)
                        if adwin_feat.drift_detected:
                            reset_on_drift(n_rows - 1, "feature")
                            continue

                # --- error-based drift check (monitor + can trigger) -------
                adwin_err.update(1 - ok)
                if adwin_err.drift_detected:
                    reset_on_drift(n_rows - 1, "error")

            print(f"  ... {n_rows:,} rows streamed, "
                  f"accuracy {accuracy.get():.4f}, "
                  f"drifts {len(drift_rows)}, snapshots {explain_count}")

    # --- Last partial window ----------------------------------------------
    if w_total > 0:
        close_window(n_rows - 1, w_phase)

    # --- Save everything before the final explanation ----------------------
    pd.DataFrame(window_rows).to_csv(
        os.path.join(RESULTS_DIR, "window_accuracy.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(
        os.path.join(RESULTS_DIR, "class_accuracy.csv"), index=False)
    pd.DataFrame(unseen_rows).to_csv(
        os.path.join(RESULTS_DIR, "unseen_accuracy.csv"), index=False)
    pd.DataFrame(drift_rows, columns=["row", "detector"]).to_csv(
        os.path.join(RESULTS_DIR, "drifts.csv"), index=False)
    pd.DataFrame(feat_scores, columns=["row", "score"]).to_csv(
        os.path.join(RESULTS_DIR, "feature_scores.csv"), index=False)

    print("\nEnd of stream — explaining final model ...")
    take_snapshot(n_rows - 1, kind="final")

    # -----------------------------------------------------------------------
    # Stability across consecutive snapshots (RQ1 + RQ2)
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
    print("RQ3 PIPELINE RESULT (MULTICLASS, 14 classes)")
    print("=" * 64)
    print(f"Rows processed      : {n_rows:,}")
    print(f"Overall accuracy    : {accuracy.get():.4f}")
    print(f"Windows             : {len(window_rows)} "
          f"({WINDOW_SIZE:,} rows each)")
    print(f"Drifts detected     : {len(drift_rows)}")
    for r, d in drift_rows:
        print(f"  drift at row {r:,} via {d} detector")
    print(f"Snapshots explained : {len(snapshots)} "
          f"(skipped: {len(skipped)})")
    for s in skipped:
        print(f"  skipped snapshot {s[0]} ({s[2]} at row {s[1]:,}): {s[3]}")
    for s in stability_rows:
        print(f"  snapshot {s['snapshot']} [{s['prev_kind']}->{s['kind']}] "
              f"{s['method']}: tau vs prev {s['kendall_tau_vs_prev']:+.3f}, "
              f"Jaccard vs prev {s['jaccard_top7_vs_prev']:.3f}")
    print(f"Time taken          : {time.time() - t0:.1f} seconds")
    print("=" * 64)

    summary_data = {
        "n_rows": n_rows,
        "overall_accuracy": accuracy.get(),
        "n_windows": len(window_rows),
        "drift_rows": drift_rows,
        "n_snapshots": len(snapshots),
        "skipped": skipped,
        "stability_rows": stability_rows,
        "window_rows": window_rows,
        "time_s": time.time() - t0,
    }
    return summary_data


if __name__ == "__main__":
    main()
