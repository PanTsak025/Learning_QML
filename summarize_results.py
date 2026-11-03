"""Turn the run CSVs of one stage into result tables (no training, runs in seconds).

    python summarize_results.py --stage first_look

Tables (printed and saved to results/main/summary_<stage>_*.csv):
  1. f1        : test macro-F1, mean +- std over seeds, per scenario x model        -> Q1, Q2
  2. ids       : binary scenarios: detection rate and false-alarm rate per model    -> Q1
  3. vqc_gap   : VQC minus the best classical model on the SAME features, paired by seed.
                 "Best classical" is picked per scenario on VALIDATION F1, never on test. -> Q2
Only test numbers are reported; the choice of comparison model uses validation numbers.
"""
import argparse, glob
import numpy as np
import pandas as pd
from run_experiments import STAGES, tasks_for, task_id

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=list(STAGES))
args = ap.parse_args()

out_dir = "results/main_smoke" if args.stage == "smoke" else "results/main"
runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out_dir}/runs_*.csv")], ignore_index=True)
tasks, fmaps = tasks_for(args.stage)
ids = {task_id(t) for t in tasks}
res = runs[runs["task"].isin(ids)].drop_duplicates(["task", "model"])

models = [f"VQC-{f.upper()}" for f in fmaps] + ["LogReg", "SVM-RBF", "RF","MLP-matched", "RF-all33"]
expected = len(ids) * len(models)
print(f"stage '{args.stage}': {len(res)} of {expected} rows present"
      + ("" if len(res) == expected else "  <-- INCOMPLETE, tables are partial"))
order = [s for s in dict.fromkeys(t["scenario"] for t in tasks)]
pd.set_option("display.width", 200)

# 1. Test macro-F1 per scenario x model (mean +- std over seeds)
g = res.groupby(["scenario", "model"])["test_macro_f1"]
f1 = (g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)).unstack()
f1 = f1.reindex(index=order, columns=models)
print("\n1) TEST macro-F1 (mean ± std over seeds)\n", f1.to_string())

# 2. Binary scenarios: detection rate (attacks caught) and false-alarm rate (benign flagged)
binary = res[res["test_detection_rate"].notna()]
if len(binary):
    ids_tab = binary.groupby(["scenario", "model"])[["test_detection_rate", "test_false_alarm_rate"]].mean().round(3)
    dr = ids_tab["test_detection_rate"].unstack().reindex(columns=models)
    far = ids_tab["test_false_alarm_rate"].unstack().reindex(columns=models)
    dr = dr.reindex([s for s in order if s in dr.index])
    far = far.reindex(dr.index)
    print("\n2a) TEST detection rate (share of attacks caught)\n", dr.to_string())
    print("\n2b) TEST false-alarm rate (share of benign traffic wrongly flagged)\n", far.to_string())

# 3. VQC vs best classical on the same features, paired by seed
classical = ["LogReg", "SVM-RBF", "RF"]
rows = []
for sc in order:
    d = res[res.scenario == sc]
    val = d[d.model.isin(classical)].groupby("model")["val_macro_f1"].mean()
    if val.empty:
        continue
    best_cls = val.idxmax()                          # chosen on VALIDATION
    cls = d[d.model == best_cls].set_index("seed")["test_macro_f1"]
    for vqc in [m for m in models if m.startswith("VQC")]:
        q = d[d.model == vqc].set_index("seed")["test_macro_f1"]
        diff = (q - cls).dropna()
        if len(diff):
            rows.append({"scenario": sc, "vqc": vqc, "best_classical": best_cls,
                         "vqc_f1": q.mean(), "classical_f1": cls.mean(),
                         "gap_mean": diff.mean(),
                         "gap_se": diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else np.nan})
gap = pd.DataFrame(rows).round(3)
if len(gap):
    print("\n3) VQC minus best classical (same features), paired by seed. "
          "Negative = classical better.\n", gap.to_string(index=False))

for name, tab in [("f1", f1), ("vqc_gap", gap)] + ([("detection", dr), ("false_alarm", far)] if len(binary) else []):
    tab.to_csv(f"{out_dir}/summary_{args.stage}_{name}.csv")