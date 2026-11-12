"""Decision rule for the diagnostic stage (run_diagnostic.py). FIXED BEFORE the stage ran.

    python analyze_diagnostic.py              # complete stage only
    python analyze_diagnostic.py --partial    # checks on an unfinished stage, no conclusions saved

Question
  Which part of the trained circuit limits the VQC: the optimiser (A1), the readout (A2) or
  the capacity (A3 more layers, A4 data re-uploading)?

Unit of analysis
  One (scenario, seed) of the five non-flood binary scenarios (Recon, Spoofing, BruteForce,
  Web, all attacks) x seeds 0-2 = 15 units. Each unit has its own training sample, so the
  change "arm minus A0" is paired within the unit.

Outcomes (both must improve; underfitting is about the fit to the TRAINING data, and the
improvement must carry over to unseen data)
  O1  fit:         training loss relative to a uniform guess (loss in bits / log2 C);
                   change = arm - A0, lower is better.
  O2  carry-over:  validation macro-F1; change = arm - A0, higher is better.
  The test set is NOT used for the decision; test macro-F1 is reported descriptively.

Rule
  An arm "explains (part of) the underfitting" when ALL hold:
   (a) O1: median change < 0 and two-sided Wilcoxon signed-rank p < 0.05 after Holm
       correction over the 4 circuit arms (A1-A4);
   (b) O2: median change > 0 and Holm-corrected p < 0.05 (same correction family size, 4);
   (c) consistency: in at least 4 of the 5 scenarios, the seed-mean change has the right sign
       for BOTH outcomes;
   (d) A2 only: its validation macro-F1 also exceeds that of R2 (same head, no circuit):
       median paired difference > 0 and Wilcoxon p < 0.05 (one extra test, uncorrected).
  Zeros are dropped before the Wilcoxon test (as in analyze_small_data.py), and a test needs
  at least 6 non-zero changes; otherwise p is reported as missing and the condition fails.
  A1 cannot have a worse training loss than A0 by construction (its starts include A0's), so
  for A1 condition (a) mainly asks whether the improvement is consistent and material; its
  size is reported.

Descriptive (no decision)
  * A5 (fidelity-kernel SVM) against A0 and against the stored SVM-RBF / best classical model
    on the same tasks: how far the encoding alone can go.
  * Training and test macro-F1 changes, parameters, CPU time, per scenario.
  * 8-class (3 seeds) per arm, same quantities.
  * Reproduction check: A0 must reproduce the stored main-study VQC-RY row of the same task
    (validation and test macro-F1, final training loss).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import run_diagnostic as rd
from run_diagnostic import DIAG, jobs, job_key
from analyze_small_data import holm

MAIN = "results/main"
CIRCUIT_ARMS = ["A1_restart", "A2_readout", "A3_deeper", "A4_reupload"]
CLASSICAL = ["LogReg", "SVM-RBF", "RF"]
ALPHA = 0.05


def load(partial):
    files = glob.glob(f"{rd.OUT}/diag_shard*.csv")
    if not files:
        raise SystemExit("no results yet: run `python run_diagnostic.py --shard i --nshards k` first")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["task", "arm", "start"])
    expected = {job_key(j) for j in jobs()}
    have = {f"{t}|{a}|{s}" for t, a, s in zip(d.task, d.arm, d.start)}
    missing = expected - have
    print(f"{len(expected) - len(missing)}/{len(expected)} jobs done")
    if missing and not partial:
        raise SystemExit("stage not finished: the decision rule runs only on the complete stage")
    return d


def arm_table(d):
    """One row per (task, arm). A1 = the start with the lowest training loss among A0's start
    and the extra starts (a choice made on training data only)."""
    rows = [d[d.arm.isin(["A0_baseline", "A2_readout", "R2_head_only", "A3_deeper", "A4_reupload", "A5_kernel"])]]
    pool = d[d.arm.isin(["A0_baseline", "A1_restart"])]
    best = pool.loc[pool.groupby("task").train_loss_bits.idxmin()].copy()
    best["arm"] = "A1_restart"
    rows.append(best)
    return pd.concat(rows, ignore_index=True)


def wtest(x):
    x = np.asarray(x, dtype=float)
    nz = x[x != 0]
    return wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan


def paired(t, arm, ref, col):
    p = t[t.arm.isin([arm, ref])].pivot_table(index=["scenario", "seed"], columns="arm", values=col)
    if arm not in p or ref not in p:      # only with --partial, before both arms have results
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["scenario", "seed"]))
    return (p[arm] - p[ref]).dropna()


def reproduction_check(t):
    files = glob.glob(f"{MAIN}/runs_*.csv")
    if not files:
        print("\n[reproduction] results/main not found: skipped")
        return None
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["task", "model"])
    a0 = t[t.arm == "A0_baseline"].set_index("task")
    s = m[(m.model == "VQC-RY") & m.task.isin(a0.index)].set_index("task")
    diff = pd.DataFrame({c: (a0[c] - s[c]).abs() for c in ["val_macro_f1", "test_macro_f1", "final_train_loss"]}).dropna()
    print(f"\n[reproduction] A0 vs stored main-study VQC-RY on {len(diff)} tasks; "
          f"largest absolute differences:\n{diff.max().round(6).to_string()}")
    # Where do the loss traces part? Identical early evaluations followed by a divergence mean
    # last-digit floating-point differences (another machine, BLAS or thread setting) that
    # COBYLA amplifies; a difference at evaluation 1 would mean different data or circuit.
    first, same = [], 0
    for tid in diff.index:
        fa = f"{rd.OUT}/curves/{tid}_A0_baseline_start0.json"
        fs = f"{MAIN}/curves/{tid}_VQC-RY.json"
        if os.path.exists(fa) and os.path.exists(fs):
            ca, cs = json.load(open(fa)), json.load(open(fs))
            k = next((i for i, (x, y) in enumerate(zip(ca, cs)) if abs(x - y) > 1e-5), None)
            first.append(len(ca) if k is None else k)
            same += k is None
    if first:
        print(f"  loss traces identical for the first {min(first)}-{max(first)} evaluations "
              f"({same} of {len(first)} identical throughout)")
    if len(diff) and diff.max().max() > 1e-3:
        print("  NOTE: A0 does not reproduce the stored runs exactly. If the traces agree at the start,"
              " this is floating-point noise amplified by COBYLA, not a change in data or model."
              " The arms are compared with A0 from THIS stage, so the decision is unaffected.")
    return m


def stored_classical(m, tasks):
    """Stored same-task classical results: MLP-matched, RF, and the best of LogReg/SVM-RBF/RF
    chosen per scenario on mean validation macro-F1 (as in analyze_small_data.py)."""
    if m is None:
        return pd.DataFrame()
    s = m[m.task.isin(tasks)]
    out = []
    for sc, g in s.groupby("scenario"):
        best = g[g.model.isin(CLASSICAL)].groupby("model").val_macro_f1.mean().idxmax()
        for name, model in [("MLP-matched", "MLP-matched"), ("RF", "RF"), (f"best classical ({best})", best)]:
            r = g[g.model == model]
            out.append({"scenario": sc, "model": name, "train_macro_f1": np.nan,
                        "val_macro_f1": r.val_macro_f1.mean(), "test_macro_f1": r.test_macro_f1.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="analyse the smoke run (code check only)")
    args = ap.parse_args()
    if args.smoke:
        rd.use_smoke_settings()
    pd.set_option("display.width", 220)
    d = load(args.partial)
    t = arm_table(d)
    m = reproduction_check(t)

    prim = t[t.scenario.isin(DIAG["primary_scenarios"])]
    rows = []
    for arm in CIRCUIT_ARMS:
        dl, dv = paired(prim, arm, "A0_baseline", "loss_ratio"), paired(prim, arm, "A0_baseline", "val_macro_f1")
        dtr, dte = paired(prim, arm, "A0_baseline", "train_macro_f1"), paired(prim, arm, "A0_baseline", "test_macro_f1")
        sl, sv = dl.groupby(level="scenario").mean(), dv.groupby(level="scenario").mean()
        rows.append({"arm": arm, "units": len(dl),
                     "loss_ratio_A0": prim[prim.arm == "A0_baseline"].loss_ratio.median(),
                     "loss_ratio_arm": prim[prim.arm == arm].loss_ratio.median(),
                     "d_loss_median": dl.median(), "p_loss": wtest(dl),
                     "d_val_median": dv.median(), "p_val": wtest(dv),
                     "d_train_f1_median": dtr.median(), "d_test_f1_median": dte.median(),
                     "scenarios_both_right": int(((sl < 0) & (sv > 0)).sum()),
                     "parameters": int(prim[prim.arm == arm].parameters.median())})
    r = pd.DataFrame(rows)
    # holm() returns its values in sorted order, keyed by the original index: align by index.
    r["p_loss_holm"] = holm(r.p_loss).reindex(r.index)
    r["p_val_holm"] = holm(r.p_val).reindex(r.index)
    a2_vs_r2 = paired(prim, "A2_readout", "R2_head_only", "val_macro_f1")
    r["A2_beats_R2_median"] = np.where(r.arm == "A2_readout", a2_vs_r2.median(), np.nan)
    r["A2_beats_R2_p"] = np.where(r.arm == "A2_readout", wtest(a2_vs_r2), np.nan)

    def verdict(x):
        ok = (x.d_loss_median < 0 and x.p_loss_holm < ALPHA and x.d_val_median > 0
              and x.p_val_holm < ALPHA and x.scenarios_both_right >= 4)
        if x.arm == "A2_readout":
            ok = ok and x.A2_beats_R2_median > 0 and x.A2_beats_R2_p < ALPHA
        return "explains (part of) the underfitting" if ok else "does not explain it"
    r["verdict"] = r.apply(verdict, axis=1)

    show = r.copy()
    for c in show.columns:
        if c.startswith("p_") or c == "A2_beats_R2_p":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2g}")
        elif show[c].dtype == float:
            show[c] = show[c].round(4)
    print(f"\nPRIMARY: arm minus A0 over {len(prim.task.unique())} (scenario, seed) units of the non-flood binary scenarios\n"
          "(loss ratio = training loss / log2 C, lower is better; F1 = macro-F1)\n", show.to_string(index=False))

    # Descriptive: per scenario and arm (mean over seeds), with the stored classical models.
    cols = ["train_macro_f1", "loss_ratio", "val_macro_f1", "test_macro_f1", "parameters", "cpu_seconds"]
    summ = t.groupby(["scenario", "arm"])[cols].mean().reset_index()
    ref = stored_classical(m, set(t.task))
    if len(ref):
        summ = pd.concat([summ, ref.rename(columns={"model": "arm"})], ignore_index=True)
    order = {sc: i for i, sc in enumerate(DIAG["scenarios"])}
    summ = summ.sort_values(["scenario", "arm"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    print("\nPer scenario, mean over seeds (stored classical rows: same tasks, from results/main)\n",
          summ.round(4).to_string(index=False))

    a5 = paired(prim, "A5_kernel", "A0_baseline", "val_macro_f1")
    print(f"\nA5 fidelity-kernel SVM minus A0, validation macro-F1: median {a5.median():+.4f}, "
          f"p = {wtest(a5):.2g} (descriptive)")
    print("\nVerdicts (rule fixed in advance):")
    for _, x in r.iterrows():
        print(f"  {x.arm:<12} {x.verdict}")
    if not args.partial:
        r.to_csv(f"{rd.OUT}/diagnostic_tests.csv", index=False)
        summ.to_csv(f"{rd.OUT}/diagnostic_summary.csv", index=False)
        print(f"\nwrote {rd.OUT}/diagnostic_tests.csv and {rd.OUT}/diagnostic_summary.csv")
