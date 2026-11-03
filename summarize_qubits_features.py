"""Summary of the qubits_features stage (RQ2 at 2/4/6/8 qubits, RQ5). Works on partial
results: every number shows how many seeds it is based on (n=1..3), and the per-scenario
setting choice is only made for scenarios whose results are complete.

    python summarize_qubits_features.py

Tables (printed and saved to results/main/summary_qubits_features_*.csv):
  1. vqc_f1   : VQC-RY test macro-F1 per scenario x (feature method, qubits)          RQ5
  2. gap      : VQC-RY minus best classical on the SAME features, paired by seed       RQ2
                (best classical chosen on VALIDATION at each setting)
  3. ceiling  : random forest on all 33 features (what the qubit bottleneck loses)
  4. mi_feats : features chosen by MI per scenario and qubit count (most common over seeds)
  5. choice   : setting per scenario for the learning curves. Rule fixed before the run:
                best mean VALIDATION macro-F1, then (one-standard-error rule) the setting
                with the FEWEST qubits within one SE of it; ties -> PCA first.
"""
import glob, json
from collections import Counter
import numpy as np
import pandas as pd
from run_experiments import tasks_for, task_id

CLASSICAL = ["LogReg", "SVM-RBF", "RF"]
OUT = "results/main"

runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
tasks, _ = tasks_for("qubits_features")
ids = {task_id(t) for t in tasks}
MODELS = ["VQC-RY", "LogReg", "SVM-RBF", "RF", "MLP-matched", "RF-all33"]
res = runs[runs["task"].isin(ids) & runs["model"].isin(MODELS)].drop_duplicates(["task", "model"]).copy()
res["setting"] = res["reducer"].str.upper() + "-" + res["n_qubits"].astype(str) + "q"
order = list(dict.fromkeys(t["scenario"] for t in tasks))
settings = [f"{r}-{q}q" for r in ["PCA", "MI"] for q in [2, 4, 6, 8]]
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(f"qubits_features: {len(res)} of {len(ids) * len(MODELS)} rows present")


def fmt(g):
    return g.mean().round(3).astype(str) + " (n=" + g.count().astype(str) + ")"


# 1. VQC F1 per scenario x setting
v = res[res.model == "VQC-RY"]
t1 = fmt(v.groupby(["scenario", "setting"])["test_macro_f1"]).unstack()
t1 = t1.reindex(index=order, columns=[s for s in settings if s in t1.columns])
print("\n1) VQC-RY TEST macro-F1 per feature method and qubits (n = seeds done)\n", t1.to_string())

# 2. Gap VQC - best classical (same features), paired by seed
rows = []
for (sc, st), d in res.groupby(["scenario", "setting"]):
    cls = d[d.model.isin(CLASSICAL)].groupby("model")["val_macro_f1"].mean()
    if cls.empty or "VQC-RY" not in set(d.model):
        continue
    best = cls.idxmax()
    gap = (d[d.model == "VQC-RY"].set_index("seed")["test_macro_f1"]
           - d[d.model == best].set_index("seed")["test_macro_f1"]).dropna()
    se = gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
    rows.append({"scenario": sc, "setting": st,
                 "gap": f"{gap.mean():+.3f}±{se:.3f} ({best[:3]}, n={len(gap)})"})
t2 = pd.DataFrame(rows).pivot(index="scenario", columns="setting", values="gap")
t2 = t2.reindex(index=[s for s in order if s in t2.index], columns=[s for s in settings if s in t2.columns])
print("\n2) VQC-RY minus best classical on the SAME features (paired by seed; negative = classical better)\n",
      t2.to_string())

# 3. Ceiling: RF on all 33 features (independent of qubits/method; mean over all its rows)
t3 = res[res.model == "RF-all33"].groupby("scenario")["test_macro_f1"].mean().round(3).reindex(order)
print("\n3) Ceiling: random forest on all 33 features (TEST macro-F1)\n", t3.to_string())

# 4. MI-selected features (most common set over seeds)
mi = v[v.reducer == "mi"]
t4 = {}
for (sc, q), d in mi.groupby(["scenario", "n_qubits"]):
    sets = [tuple(json.loads(f)) for f in d["features"] if isinstance(f, str) and f != "[]"]
    if sets:
        common, cnt = Counter(sets).most_common(1)[0]
        t4[(sc, f"{q}q")] = ", ".join(common) + f"  [{cnt}/{len(sets)} seeds]"
if t4:
    t4 = pd.Series(t4).unstack().reindex([s for s in order if s in {k[0] for k in t4}])
    print("\n4) Features selected by MI (most common over seeds)\n", t4.to_string())

# 5. Setting per scenario for the learning curves - complete scenarios only.
#    Rule (aligned with the ansatz pilot): one-standard-error rule on PAIRED validation
#    differences. Candidates: settings with <= MAX_QUBITS qubits (cap decided before the
#    learning curves: 8 qubits costs ~13x a 4-qubit training). Best = highest mean validation
#    macro-F1; a setting is equivalent if its mean gap to the best, paired by seed (same
#    training sample), is <= the SE of that gap; choose the fewest qubits, then PCA.
#    The original (unpaired) rule's pick is reported next to it for transparency.
MAX_QUBITS = 6
n_settings = {sc: len({task_id(t) for t in tasks if t["scenario"] == sc}) for sc in order}


def pick(cands):
    cands = sorted(cands, key=lambda s: (int(s.split("-")[1][:-1]), not s.startswith("PCA")))
    return cands[0]


rows = []
for sc in order:
    d = v[v.scenario == sc]
    if len(d) < n_settings[sc]:
        rows.append({"scenario": sc, "chosen_setting": f"incomplete ({len(d)}/{n_settings[sc]} trainings)"})
        continue
    val = d.pivot_table(index="seed", columns="setting", values="val_macro_f1")
    capped = [c for c in val.columns if int(c.split("-")[1][:-1]) <= MAX_QUBITS]
    best = val[capped].mean().idxmax()
    equivalent = []
    for c in capped:
        gap = val[best] - val[c]
        se = gap.std() / np.sqrt(len(gap))
        if c == best or gap.mean() <= se:
            equivalent.append(c)
    chosen = pick(equivalent)
    # original rule: SE of the best setting alone, all qubit counts
    m, s_ = val.mean(), val.std() / np.sqrt(len(val))
    b0 = m.idxmax()
    original = pick([c for c in val.columns if m[c] >= m[b0] - s_[b0]])
    rows.append({"scenario": sc, "best_by_val": f"{best} ({val[best].mean():.3f})",
                 "chosen_setting": chosen, "chosen_val_f1": round(val[chosen].mean(), 3),
                 "equivalent_settings": ", ".join(sorted(equivalent)),
                 "original_rule_pick": original})
t5 = pd.DataFrame(rows)
print(f"\n5) Setting per scenario for the learning curves (paired one-SE rule on VALIDATION F1, "
      f"<= {MAX_QUBITS} qubits, fewest qubits then PCA)\n", t5.to_string(index=False))

for name, tab in [("vqc_f1", t1), ("gap", t2), ("ceiling", t3), ("choice", t5)]:
    tab.to_csv(f"{OUT}/summary_qubits_features_{name}.csv")