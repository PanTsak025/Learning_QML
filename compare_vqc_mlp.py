"""VQC vs a classical neural network of the SAME size (parameter-matched MLP).

    python compare_vqc_mlp.py

Both models get the same features, the same training samples and the same test set; the
MLP has (as close as possible) the same number of trainable parameters as the VQC.
Every comparison is paired by seed (same training sample) and reported on TEST macro-F1:
    gap = VQC - MLP, mean +- SE over seeds.
Verdict per cell (same one-standard-error rule as everywhere else in the study):
    "="  |gap| <= SE    -> the data cannot tell them apart
    "Q"  gap  >  SE     -> VQC better
    "M"  gap  < -SE     -> MLP better

Tables (printed and saved to results/main/vqc_vs_mlp_*.csv):
  1. learning curves : scenario x sampling x training size
  2. qubits/features : scenario x (PCA/MI, 2/4/6/8 qubits) at 200 per class
  3. tally of verdicts per table
  4. parameter counts actually used (VQC vs MLP)
"""
import glob
import numpy as np
import pandas as pd
from run_experiments import tasks_for, task_id

OUT = "results/main"
runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
runs = runs[runs.model.isin(["VQC-RY", "MLP-matched"])].drop_duplicates(["task", "model"])
pd.set_option("display.width", 250, "display.max_colwidth", 40)


def cell(d):
    p = d.pivot_table(index="seed", columns="model", values="test_macro_f1")
    if not {"VQC-RY", "MLP-matched"} <= set(p.columns):
        return None, None
    gap = (p["VQC-RY"] - p["MLP-matched"]).dropna()
    se = gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
    g = gap.mean()
    verdict = "=" if not (abs(g) > se) else ("Q" if g > 0 else "M")
    return f"{g:+.3f}±{se:.3f} {verdict}", verdict


def table(stage, row_keys, col_key, col_order):
    ids = {task_id(t) for t in tasks_for(stage)[0]}
    d = runs[runs.task.isin(ids)].copy()
    if "setting" in (col_key,):
        d["setting"] = d["reducer"].str.upper() + "-" + d["n_qubits"].astype(str) + "q"
    cells, verdicts = {}, []
    for key, g in d.groupby(row_keys + [col_key]):
        txt, v = cell(g)
        if txt:
            cells[key] = txt
            verdicts.append(v)
    t = pd.Series(cells).unstack()
    t = t.reindex(columns=[c for c in col_order if c in t.columns])
    return t, pd.Series(verdicts).value_counts()


scen_order = list(dict.fromkeys(t["scenario"] for t in tasks_for("learning_curves")[0]))

lc, lc_tally = table("learning_curves", ["scenario", "sampling"], "n_train", [25, 50, 100, 200, 400, 800])
lc = lc.reindex([(s, m) for s in scen_order for m in ["random", "kmeans"] if (s, m) in lc.index])
print("1) Learning curves: VQC - MLP (same parameters, same features), TEST macro-F1, paired by seed\n"
      "   = tie (within 1 SE)   Q = VQC better   M = MLP better\n", lc.to_string())

settings = [f"{r}-{q}q" for r in ["PCA", "MI"] for q in [2, 4, 6, 8]]
qf, qf_tally = table("qubits_features", ["scenario"], "setting", settings)
qf = qf.reindex([s for s in scen_order if s in qf.index])
print("\n2) Qubits/features at 200 per class: VQC - MLP\n", qf.to_string())

tally = pd.DataFrame({"learning_curves": lc_tally, "qubits_features": qf_tally}).fillna(0).astype(int)
tally.index = tally.index.map({"=": "tie (=)", "Q": "VQC better (Q)", "M": "MLP better (M)"})
print("\n3) Tally of verdicts\n", tally.to_string())

par = runs.assign(type=np.where(runs.scenario == "multiclass", "8-class", "binary")) \
          .groupby(["type", "n_qubits", "model"])["parameters"].first().unstack()
print("\n4) Trainable parameters actually used\n", par.to_string())

lc.to_csv(f"{OUT}/vqc_vs_mlp_learning_curves.csv")
qf.to_csv(f"{OUT}/vqc_vs_mlp_qubits_features.csv")
tally.to_csv(f"{OUT}/vqc_vs_mlp_tally.csv")