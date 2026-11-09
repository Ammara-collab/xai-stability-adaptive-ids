"""
RQ3 analysis — verifies every number from the result CSVs (recomputed,
not copied from the console log) and writes RESULTS_SUMMARY_RQ3.md.

Checks:
  1. Overall + per-window + per-class accuracy (any real drops?)
  2. Drifts by detector (feature vs error) and where they fired
  3. Feature-drift score behaviour around the file switches
  4. SHAP/LIME stability means, overall + Phase A vs Phase B
  5. RQ3 TEST: correlation (Pearson + Spearman) between per-snapshot
     stability and accuracy at that snapshot (and accuracy change)
  6. Unseen (frozen) model: Phase A vs Phase B accuracy
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results_rq3")

w = pd.read_csv(os.path.join(R, "window_accuracy.csv"))
c = pd.read_csv(os.path.join(R, "class_accuracy.csv"))
s = pd.read_csv(os.path.join(R, "cycle_stability.csv"))
d = pd.read_csv(os.path.join(R, "drifts.csv"))
f = pd.read_csv(os.path.join(R, "feature_scores.csv"))
u = pd.read_csv(os.path.join(R, "unseen_accuracy.csv"))

out = []
def line(t=""):
    out.append(t)
    print(t)

line("=" * 68)
line("RQ3 ANALYSIS — recomputed from CSVs")
line("=" * 68)

# ---------------------------------------------------------------- 1. accuracy
n_rows = int(w["n"].sum())
wacc = float((w["accuracy"] * w["n"]).sum() / w["n"].sum())
line(f"\n[1] ACCURACY (main multiclass model, 14 classes)")
line(f"    rows: {n_rows:,}  windows: {len(w)}  weighted accuracy: {wacc:.6f}")
line(f"    windows with accuracy < 1.0: {(w['accuracy'] < 1.0).sum()}")
line(f"    windows with accuracy < 0.999: {(w['accuracy'] < 0.999).sum()}")
line(f"    windows with accuracy < 0.997: {(w['accuracy'] < 0.997).sum()}")
line("    worst 8 windows:")
for _, r in w.nsmallest(8, "accuracy").iterrows():
    line(f"      window {int(r['window'])} (rows {int(r['start_row'])}-"
         f"{int(r['end_row'])}, phase {r['phase']}): acc {r['accuracy']:.4f}, "
         f"unseen {r['acc_unseen']:.4f}")

# per-class accuracy overall
line(f"\n    PER-CLASS accuracy (whole stream):")
cls = c.groupby("class_name").apply(
    lambda g: pd.Series({
        "n": g["n"].sum(),
        "acc": (g["correct"].sum() / g["n"].sum()),
        "min_window_acc": g["accuracy"].min(),
        "n_windows_below_0.99": (g["accuracy"] < 0.99).sum(),
    }), include_groups=False)
for name, r in cls.iterrows():
    line(f"      {name:20s} n={int(r['n']):7,} acc={r['acc']:.4f} "
         f"min_window={r['min_window_acc']:.4f} "
         f"windows<0.99: {int(r['n_windows_below_0.99'])}")

# ---------------------------------------------------------------- 2. drifts
line(f"\n[2] DRIFTS: {len(d)} total")
line(f"    by detector: {d['detector'].value_counts().to_dict()}")
for _, r in d.iterrows():
    line(f"      row {int(r['row']):,} via {r['detector']}")
# file switch rows (0-based end of each file, from the family summary):
# Benign ends 362,360; floods end 2,260,936; Recon starts 2,260,937.
switches = {"Benign->DDoS": 362361, "DDoS->Recon": 2260937}
for name, row in switches.items():
    near = d[(d["row"] >= row - 2000) & (d["row"] <= row + 2000)]
    line(f"    switch {name} (row {row:,}): drifts within +-2000 rows: "
         f"{len(near)} {near['detector'].tolist() if len(near) else ''}")

# ------------------------------------------------- 3. feature drift scores
line(f"\n[3] FEATURE-DRIFT SCORE (mean standardised shift, every 100 rows)")
line(f"    score rows: {len(f)}  overall mean {f['score'].mean():.4f} "
     f"median {f['score'].median():.4f} max {f['score'].max():.4f}")
for name, row in switches.items():
    seg = f[(f["row"] >= row - 1000) & (f["row"] <= row + 1000)]
    before = f[(f["row"] >= row - 5000) & (f["row"] < row - 1000)]
    line(f"    {name}: mean score +-1000 rows {seg['score'].mean():.4f} "
         f"(max {seg['score'].max():.4f}) vs 5k rows before "
         f"{before['score'].mean():.4f}")

# ---------------------------------------------------------------- 4. stability
line(f"\n[4] STABILITY (snapshots: {s['snapshot'].nunique()} unique, "
      f"{len(s)} rows = {s['snapshot'].nunique()} snapshots x 2 methods)")
for method in ("SHAP", "LIME"):
    sub = s[s["method"] == method]
    line(f"    {method}: mean tau vs prev {sub['kendall_tau_vs_prev'].mean():+.3f}, "
         f"mean Jaccard vs prev {sub['jaccard_top7_vs_prev'].mean():.3f}, "
         f"min tau {sub['kendall_tau_vs_prev'].min():+.3f}, "
         f"min Jaccard {sub['jaccard_top7_vs_prev'].min():.3f}")
# Phase split: snapshots with end_row < 2,260,937 are Phase A (+benign)
s["phase"] = np.where(s["end_row"] < 2260937, "A", "B")
for phase in ("A", "B"):
    for method in ("SHAP", "LIME"):
        sub = s[(s["phase"] == phase) & (s["method"] == method)]
        line(f"    Phase {phase} {method} (n={len(sub)}): tau "
             f"{sub['kendall_tau_vs_prev'].mean():+.3f}, Jaccard "
             f"{sub['jaccard_top7_vs_prev'].mean():.3f}")

# ------------------------------------------------------------- 5. RQ3 TEST
line(f"\n[5] RQ3 TEST: stability vs accuracy at each snapshot")
# join each snapshot to its containing window (window.start_row <= end_row)
snap = s.drop_duplicates("snapshot").copy()
acc_at = []
for _, r in snap.iterrows():
    win = w[(w["start_row"] <= r["end_row"]) & (w["end_row"] >= r["end_row"])]
    acc_at.append(win["accuracy"].iloc[0] if len(win) else np.nan)
snap["window_accuracy"] = acc_at
snap["acc_change"] = snap["window_accuracy"].diff()
line(f"    snapshots with a joined window accuracy: "
     f"{snap['window_accuracy'].notna().sum()}/{len(snap)}")
line(f"    snapshot-window accuracy: mean {snap['window_accuracy'].mean():.6f}, "
     f"min {snap['window_accuracy'].min():.4f}, "
     f"std {snap['window_accuracy'].std():.6f}, "
     f"n unique values: {snap['window_accuracy'].nunique()}")
for method in ("SHAP", "LIME"):
    sub = s[s["method"] == method].merge(
        snap[["snapshot", "window_accuracy", "acc_change"]],
        on="snapshot", how="left")
    for metric in ("kendall_tau_vs_prev", "jaccard_top7_vs_prev"):
        x = sub[metric].values
        y = sub["window_accuracy"].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3 and np.std(y[ok]) > 0:
            pr, pp = pearsonr(x[ok], y[ok])
            sr, sp = spearmanr(x[ok], y[ok])
            line(f"    {method} {metric} vs window accuracy: Pearson r={pr:+.3f} "
                 f"(p={pp:.3f}), Spearman rho={sr:+.3f} (p={sp:.3f}), n={ok.sum()}")
        else:
            line(f"    {method} {metric} vs window accuracy: NOT COMPUTABLE "
                 f"(accuracy std={np.std(y[ok]):.6f}, n={ok.sum()})")
        y2 = sub["acc_change"].values
        ok2 = np.isfinite(x) & np.isfinite(y2)
        if ok2.sum() > 3 and np.std(y2[ok2]) > 0:
            pr2, pp2 = pearsonr(x[ok2], y2[ok2])
            sr2, sp2 = spearmanr(x[ok2], y2[ok2])
            line(f"    {method} {metric} vs accuracy CHANGE: Pearson r={pr2:+.3f} "
                 f"(p={pp2:.3f}), Spearman rho={sr2:+.3f} (p={sp2:.3f})")
        else:
            line(f"    {method} {metric} vs accuracy CHANGE: NOT COMPUTABLE "
                 f"(std={np.std(y2[ok2]):.6f})")
# lowest-stability snapshots: what was accuracy there?
line("    lowest-stability snapshots and their window accuracy:")
low = s.nsmallest(6, "jaccard_top7_vs_prev")[
    ["snapshot", "method", "kendall_tau_vs_prev", "jaccard_top7_vs_prev",
     "end_row"]].merge(snap[["snapshot", "window_accuracy"]], on="snapshot")
for _, r in low.iterrows():
    line(f"      snapshot {int(r['snapshot'])} {r['method']}: tau "
         f"{r['kendall_tau_vs_prev']:+.3f}, Jaccard {r['jaccard_top7_vs_prev']:.3f}, "
         f"window accuracy {r['window_accuracy']:.4f}")

# ------------------------------------------------------------- 6. unseen model
line(f"\n[6] UNSEEN MODEL (frozen after Phase A: trained on Benign + 8 floods)")
uA = u[u["phase"].isin(["benign", "A"])]
uB = u[u["phase"] == "B"]
line(f"    Phase A (learning, prequential): mean acc "
     f"{uA['accuracy'].mean():.4f}, min {uA['accuracy'].min():.4f}")
line(f"    Phase B (FROZEN, unseen Recon): mean acc "
     f"{uB['accuracy'].mean():.4f}, min {uB['accuracy'].min():.4f}, "
     f"max {uB['accuracy'].max():.4f}")
line(f"    Phase B windows with unseen acc == 0: {(uB['accuracy'] == 0).sum()}"
     f"/{len(uB)}")
# main model accuracy on the same Phase B windows (for contrast)
wB = w[w["phase"] == "B"]
line(f"    Main model (kept learning) Phase B mean acc: "
     f"{wB['accuracy'].mean():.4f}, min {wB['accuracy'].min():.4f}")

# save the joined snapshot table for the summary
piv = s.pivot_table(index="snapshot", columns="method",
                    values=["kendall_tau_vs_prev", "jaccard_top7_vs_prev"])
piv.columns = [f"{m.lower()}_{v.replace('kendall_tau_vs_prev', 'tau').replace('jaccard_top7_vs_prev', 'jaccard')}"
               for v, m in piv.columns]
snap_out = snap[["snapshot", "cycle", "kind", "end_row",
                 "window_accuracy"]].merge(
    piv, left_on="snapshot", right_index=True, how="left")
snap_out.to_csv(os.path.join(R, "rq3_snapshot_table.csv"), index=False)
line(f"\nSaved rq3_snapshot_table.csv ({len(snap_out)} snapshots)")
line("=" * 68)

# stash key numbers for the summary writer
import json
summary_numbers = {
    "n_rows": n_rows,
    "n_windows": len(w),
    "weighted_accuracy": wacc,
    "windows_below_1": int((w["accuracy"] < 1.0).sum()),
    "windows_below_0.999": int((w["accuracy"] < 0.999).sum()),
    "windows_below_0.997": int((w["accuracy"] < 0.997).sum()),
    "min_window_accuracy": float(w["accuracy"].min()),
    "n_drifts": len(d),
    "drifts_by_detector": {k: int(v) for k, v in
                           d["detector"].value_counts().items()},
    "n_snapshots": int(s["snapshot"].nunique()),
    "unseen_phaseB_mean": float(uB["accuracy"].mean()),
    "unseen_phaseB_max": float(uB["accuracy"].max()),
    "unseen_phaseB_zero_windows": int((uB["accuracy"] == 0).sum()),
    "unseen_phaseB_n": len(uB),
    "main_phaseB_mean": float(wB["accuracy"].mean()),
    "main_phaseB_min": float(wB["accuracy"].min()),
    "snapshot_acc_mean": float(snap["window_accuracy"].mean()),
    "snapshot_acc_min": float(snap["window_accuracy"].min()),
    "snapshot_acc_std": float(snap["window_accuracy"].std()),
    "snapshot_acc_n_unique": int(snap["window_accuracy"].nunique()),
}
with open(os.path.join(R, "rq3_numbers.json"), "w") as fh:
    json.dump(summary_numbers, fh, indent=2)
line("Saved rq3_numbers.json")
