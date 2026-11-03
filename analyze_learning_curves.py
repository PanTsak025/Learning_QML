"""RQ3/RQ4: how much training data does the VQC need to be 'good' as an IDS?

    python analyze_learning_curves.py            # after the learning_curves stage has finished

"Good" has two definitions, both FIXED BEFORE the learning-curve run (no new thresholds:
both use the one-standard-error rule used for every other decision in this study).
Computed per scenario and per sampling strategy (random / kmeans), on TEST macro-F1
(reported numbers; no model is selected with them). Seeds give the spread:
mean and standard error (SE = std / sqrt(n_seeds)) at every training size.

  A. MATCHES CLASSICAL: the smallest training size n_A from which the VQC is no worse
     than the best classical model on the SAME features by more than one SE, at that size
     AND at every larger size (so a single lucky size does not count). The gap is paired
     by seed (same training sample). "Best classical" at each size = the classical model
     with the highest mean VALIDATION macro-F1 at that size.
     -> "QML needs n_A samples per class to match classical" (or "not within 800").

  B. REACHES ITS PLATEAU: the smallest training size n_B from which the model's mean F1
     stays within one SE of its own best mean F1 (SE taken at the best size), at that size
     and every larger size. Computed for the VQC AND for each classical model, so the
     data needs can be compared ("VQC plateaus at n_B, SVM at m_B").

RQ4: n_A and n_B for kmeans vs random show whether representative sampling reduces the
data needed.
"""
import glob
import numpy as np
import pandas as pd
from run_experiments import tasks_for, task_id

CLASSICAL = ["LogReg", "SVM-RBF", "RF"]


def from_size_onwards(sizes, ok):
    """Smallest size s such that ok holds at s and every larger size; None if never."""
    result = None
    for s, good in sorted(zip(sizes, ok), reverse=True):
        if not good:
            break
        result = s
    return result


def plateau_size(curve):
    """Definition B. curve: DataFrame indexed by size with columns mean, se."""
    best = curve["mean"].idxmax()
    threshold = curve.loc[best, "mean"] - curve.loc[best, "se"]
    n_b = from_size_onwards(curve.index, curve["mean"] >= threshold)
    if n_b is None:   # the largest size falls below the plateau band: the curve dips at the end
        n_b = f"no stable plateau (best at {best})"
    return n_b, round(curve.loc[best, "mean"], 3)


def analyse(res, vqc_name="VQC-RY"):
    rows_a, rows_b = [], []
    for (sc, samp), d in res.groupby(["scenario", "sampling"]):
        sizes = sorted(d["n_train"].unique())
        # ---- Definition A: VQC vs best classical (chosen on validation), paired by seed ----
        ok, details = [], []
        for n in sizes:
            dn = d[d.n_train == n]
            best_cls = dn[dn.model.isin(CLASSICAL)].groupby("model")["val_macro_f1"].mean().idxmax()
            q = dn[dn.model == vqc_name].set_index("seed")["test_macro_f1"]
            c = dn[dn.model == best_cls].set_index("seed")["test_macro_f1"]
            gap = (q - c).dropna()
            se = gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
            ok.append(bool(gap.mean() >= -se))
            details.append(f"{n}:{gap.mean():+.3f}±{se:.3f}({best_cls})")
        n_a = from_size_onwards(sizes, ok)
        rows_a.append({"scenario": sc, "sampling": samp,
                       "n_A_matches_classical": n_a if n_a is not None else f"not within {max(sizes)}",
                       "gap_per_size": "  ".join(details)})
        # ---- Definition B: plateau size for the VQC and every classical model ----
        row = {"scenario": sc, "sampling": samp}
        for m in [vqc_name] + CLASSICAL + ["MLP-matched"]:
            g = d[d.model == m].groupby("n_train")["test_macro_f1"]
            curve = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count())})
            n_b, best = plateau_size(curve)
            row[f"n_B_{m}"] = n_b
            row[f"best_F1_{m}"] = best
        rows_b.append(row)
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


if __name__ == "__main__":
    out_dir = "results/main"
    runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out_dir}/runs_*.csv")], ignore_index=True)
    tasks, fmaps = tasks_for("learning_curves")
    res = runs[runs["task"].isin({task_id(t) for t in tasks})].drop_duplicates(["task", "model"])
    print(f"learning_curves rows used: {len(res)}")
    a, b = analyse(res)
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print("\nA) smallest training size (per class) from which the VQC matches the best classical "
          "model on the same features (within 1 SE, paired by seed)\n", a.to_string(index=False))
    print("\nB) smallest training size (per class) from which each model stays within 1 SE of its "
          "own best F1 (plateau)\n", b.to_string(index=False))
    a.to_csv(f"{out_dir}/learning_curves_A_match_classical.csv", index=False)
    b.to_csv(f"{out_dir}/learning_curves_B_plateau.csv", index=False)