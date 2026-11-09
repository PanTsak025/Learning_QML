"""Confirmatory test for the small_data stage. Everything below was FIXED BEFORE the stage ran.

    python run_experiments.py --stage small_data --shard i --nshards 8     # i = 0..7
    python analyze_small_data.py

Why this follow-up exists
  In the learning curves (seeds 0-2), on the window-matched scenarios at <= 50 samples per
  class, the VQC was ahead of the parameter-matched MLP in 12 of 16 cells. That pattern was
  found AFTER looking at the results, so it is only a hypothesis. This stage tests it on new
  training samples (seeds 3-12) that played no part in suggesting it.

Hypothesis (H1)
  On window-matched scenarios (Recon, Spoofing, BruteForce, Web; same 10-packet extraction
  window as Benign) with 10, 25 or 50 training samples per class, the VQC's test macro-F1 is
  higher than that of the MLP with (as close as possible) the same number of parameters.

Unit of analysis
  One (scenario, seed): the VQC-minus-MLP test macro-F1 gap averaged over the three training
  sizes. Different seeds draw different training samples; the three sizes of one seed are
  nested (10 within 25 within 50), so they are averaged rather than counted as separate
  observations. 4 scenarios x 10 seeds = 40 values.

Primary test (decides H1)
  Two-sided Wilcoxon signed-rank test on the 40 gaps, alpha = 0.05.
  H1 is CONFIRMED only if p < 0.05 AND the median gap is > 0. Otherwise it is NOT confirmed,
  and the paper reports that.

Secondary results (descriptive, Holm-corrected within each family)
  * per scenario: Wilcoxon on its 10 seed gaps (4 tests);
  * per training size: Wilcoxon on 40 (scenario, seed) gaps at that size (3 tests);
  * the same primary test against the best tuned classical model (LogReg, SVM-RBF or RF;
    chosen per (scenario, size) on mean VALIDATION macro-F1, as in the rest of the study).
  Each line also gives the median gap, the mean gap with a 95% bootstrap interval
  (10,000 resamples, seed 0), and how many units the VQC is ahead / behind.
"""
import argparse, glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from run_experiments import tasks_for, task_id

OUT = "results/main"
STAGE = "small_data"
CLASSICAL = ["LogReg", "SVM-RBF", "RF"]
ALPHA = 0.05


def load():
    files = glob.glob(f"{OUT}/runs_{STAGE}_shard*.csv")
    if not files:
        raise SystemExit(f"no results yet: run `python run_experiments.py --stage {STAGE}` first")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["task", "model"])
    ids = {task_id(t) for t in tasks_for(STAGE)[0]}
    return runs[runs.task.isin(ids)], len(ids)


def gaps(runs, other):
    """VQC minus `other` per (scenario, n_train, seed); `other` = a model name or 'best_classical'."""
    rows = []
    for (sc, n), g in runs.groupby(["scenario", "n_train"]):
        comp = other
        if other == "best_classical":
            comp = g[g.model.isin(CLASSICAL)].groupby("model")["val_macro_f1"].mean().idxmax()
        p = g.pivot_table(index="seed", columns="model", values="test_macro_f1")
        if "VQC-RY" not in p or comp not in p:
            continue
        for seed, v in (p["VQC-RY"] - p[comp]).dropna().items():
            rows.append({"scenario": sc, "n_train": n, "seed": seed, "gap": v, "comparator": comp})
    return pd.DataFrame(rows)


def test(x, rng):
    x = np.asarray(x, dtype=float)
    nz = x[x != 0]
    p = wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan
    boot = rng.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
    return {"units": len(x), "median_gap": np.median(x), "mean_gap": x.mean(),
            "ci95_low": np.percentile(boot, 2.5), "ci95_high": np.percentile(boot, 97.5),
            "VQC_ahead": int((x > 0).sum()), "VQC_behind": int((x < 0).sum()), "p": p}


def holm(ps):
    ps = pd.Series(ps, dtype=float)
    order = ps.sort_values().index
    m, running, adj = ps.notna().sum(), 0.0, {}
    for i, k in enumerate(order):
        if pd.isna(ps[k]):
            adj[k] = np.nan
            continue
        running = max(running, min(1.0, (m - i) * ps[k]))
        adj[k] = running
    return pd.Series(adj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="allow an unfinished stage (checks only)")
    args = ap.parse_args()
    runs, n_tasks = load()
    done = runs[runs.model == "VQC-RY"].task.nunique()
    print(f"{done}/{n_tasks} tasks with a trained VQC")
    if done < n_tasks and not args.partial:
        raise SystemExit("stage not finished: the confirmatory test runs only on the complete stage")
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)
    results = []
    for other, label in [("MLP-matched", "MLP-matched"), ("best_classical", "best classical")]:
        g = gaps(runs, other)
        unit = g.groupby(["scenario", "seed"])["gap"].mean()          # average the nested sizes
        results.append({"comparison": f"VQC vs {label}", "subset": "PRIMARY: all scenarios, sizes averaged",
                        **test(unit, rng)})
        fam = [{"comparison": f"VQC vs {label}", "subset": f"scenario = {sc}", **test(u, rng)}
               for sc, u in unit.groupby(level="scenario")]
        fam += [{"comparison": f"VQC vs {label}", "subset": f"n_train = {n}", **test(u["gap"], rng)}
                for n, u in g.groupby("n_train")]
        part = pd.DataFrame(fam)
        part["p_holm"] = pd.concat([holm(part[part.subset.str.startswith(k)]["p"])
                                    for k in ["scenario", "n_train"]])
        results += part.to_dict("records")
        if other == "best_classical":
            print("\nbest classical model chosen per (scenario, size) on validation:\n",
                  g.groupby(["scenario", "n_train"])["comparator"].first().unstack().to_string())
    r = pd.DataFrame(results)
    show = r.copy()
    for c in ["median_gap", "mean_gap", "ci95_low", "ci95_high"]:
        show[c] = show[c].round(4)
    for c in ["p", "p_holm"]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2g}")
    print("\nVQC minus comparator, test macro-F1 (unit = one scenario x seed, sizes averaged,"
          " unless the subset is a training size)\n", show.to_string(index=False))
    prim = r[(r.comparison == "VQC vs MLP-matched") & r.subset.str.startswith("PRIMARY")].iloc[0]
    confirmed = prim.p < ALPHA and prim.median_gap > 0
    tag = " [PARTIAL CHECK ONLY, not the confirmatory result]" if done < n_tasks else ""
    print(f"\nH1 (VQC > same-size MLP with <= 50 samples per class, window-matched scenarios){tag}: "
          f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'} "
          f"(median gap {prim.median_gap:+.4f}, p = {prim.p:.3g}, {prim.VQC_ahead} ahead / {prim.VQC_behind} behind)")
    if not args.partial:
        r.to_csv(f"{OUT}/small_data_tests.csv", index=False)
        print(f"wrote {OUT}/small_data_tests.csv")
