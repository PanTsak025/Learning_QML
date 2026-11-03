"""Best quantum vs best classical, side by side (first_look + qubits_features results).

    python compare_best.py

Fair-selection rule: "best" is always chosen on VALIDATION macro-F1 (mean over seeds);
the TEST macro-F1 of the chosen model is what is reported. Classical models only ever get
the SAME features and samples as the VQC (RF-all33 is shown separately as the ceiling).

Table 1 (per scenario x setting): best quantum model vs best classical model at that
        exact setting (same features), gap paired by seed.
Table 2 (per scenario, overall): the best quantum configuration over ALL settings vs the
        best classical configuration over ALL settings, plus the all-33-feature ceiling.
"""
import glob
import numpy as np
import pandas as pd
from run_experiments import tasks_for, task_id

QUANTUM = ["VQC-RY", "VQC-ZZ"]
CLASSICAL = ["LogReg", "SVM-RBF", "RF", "MLP-matched"]
OUT = "results/main"

ids = set()
for stage in ["first_look", "qubits_features"]:
    ids |= {task_id(t) for t in tasks_for(stage)[0]}
runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
res = runs[runs.task.isin(ids)].drop_duplicates(["task", "model"]).copy()
res["setting"] = res["reducer"].str.upper() + "-" + res["n_qubits"].astype(str) + "q"
order = list(dict.fromkeys(t["scenario"] for t in tasks_for("qubits_features")[0]))
settings = [f"{r}-{q}q" for r in ["PCA", "MI"] for q in [2, 4, 6, 8]]
pd.set_option("display.width", 250, "display.max_colwidth", 60)


def best_of(d, models):
    """Model (among `models`) with the highest mean VALIDATION F1 in d; its test F1 per seed."""
    cand = d[d.model.isin(models)]
    if cand.empty:
        return None, None
    m = cand.groupby("model")["val_macro_f1"].mean().idxmax()
    return m, cand[cand.model == m].set_index("seed")["test_macro_f1"]


def paired(q, c):
    gap = (q - c).dropna()
    se = gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
    return gap.mean(), se, len(gap)


# ---- Table 1: per scenario x setting ----
rows = []
for (sc, st), d in res.groupby(["scenario", "setting"]):
    qm, q = best_of(d, QUANTUM)
    cm, c = best_of(d, CLASSICAL)
    if qm is None or cm is None:
        continue
    g, se, n = paired(q, c)
    rows.append({"scenario": sc, "setting": st,
                 "best_quantum": f"{q.mean():.3f} {qm}",
                 "best_classical": f"{c.mean():.3f} {cm}",
                 "gap (Q-C)": f"{g:+.3f}±{se:.3f}" + ("" if n == 3 else f" (n={n})")})
t1 = pd.DataFrame(rows)
t1["scenario"] = pd.Categorical(t1["scenario"], order)
t1["setting"] = pd.Categorical(t1["setting"], settings)
t1 = t1.sort_values(["scenario", "setting"])
print("1) Best quantum vs best classical at each setting (same features; chosen on validation, "
      "TEST macro-F1; gap paired by seed)\n")
for sc, d in t1.groupby("scenario", observed=True):
    print(f"--- {sc}\n", d.drop(columns="scenario").to_string(index=False), "\n")

# ---- Table 2: overall best per scenario ----
rows = []
for sc in order:
    d = res[res.scenario == sc]
    if d.empty:
        continue
    def overall(models):
        c = d[d.model.isin(models)].groupby(["setting", "model"])
        val = c["val_macro_f1"].mean()
        if val.empty:
            return None
        (st, m) = val.idxmax()
        sel = d[(d.setting == st) & (d.model == m)]["test_macro_f1"]
        return st, m, sel.mean(), sel.std() / np.sqrt(len(sel)), len(sel)
    q, c = overall(QUANTUM), overall(CLASSICAL)
    ceiling = d[d.model == "RF-all33"]["test_macro_f1"].mean()
    rows.append({"scenario": sc,
                 "best_quantum": f"{q[2]:.3f}±{q[3]:.3f}  {q[1]} {q[0]}" if q else "",
                 "best_classical (same-feature)": f"{c[2]:.3f}±{c[3]:.3f}  {c[1]} {c[0]}" if c else "",
                 "difference (Q-C)": f"{q[2] - c[2]:+.3f}" if q and c else "",
                 "RF all 33 (ceiling)": round(ceiling, 3)})
t2 = pd.DataFrame(rows)
print("2) Overall: best quantum configuration vs best classical configuration per scenario "
      "(each chosen on validation over all settings; TEST macro-F1 ± SE)\n", t2.to_string(index=False))
t1.to_csv(f"{OUT}/compare_best_per_setting.csv", index=False)
t2.to_csv(f"{OUT}/compare_best_overall.csv", index=False)