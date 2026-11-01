"""Ansatz pilot: which trainable circuit (family and depth) to use, decided on VALIDATION data.

    python ansatz_pilot.py --smoke   # ~2 minute check that everything runs
    python ansatz_pilot.py           # the real pilot (~1-2 hours; resumable)

Same setup as the optimizer pilot, so results are comparable; only the ansatz varies:
  scenario benign_vs_attack, 4 qubits, PCA, 100 training / 500 validation samples per
  class, both feature maps, 3 seeds, COBYLA (chosen by the optimizer pilot).
Candidates: {RealAmplitudes, EfficientSU2} x reps {1, 2, 3}, linear entanglement
(motivated by the expressibility table in results/ansatz_expressibility.csv).
The test set is never touched.
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from qiskit_machine_learning.optimizers import COBYLA

from qml_data import load_scenario, load_validation, to_qubits
from feature_maps import ANGLE_MAX
from ansatz import build_ansatz, effective_parameters
from vqc_model import build_vqc

warnings.filterwarnings("ignore", category=UserWarning)   # COBYLA prints a harmless MAXFUN note

SMOKE = "--smoke" in sys.argv
PILOT = {
    "scenario": "benign_vs_attack", "n_classes": 2,
    "feature_maps": ["ry", "zz"],
    "ansatz_candidates": [(a, r) for a in ["real_amplitudes", "efficient_su2"] for r in [1, 2, 3]],
    "n_qubits": 4, "reducer": "pca",
    "n_train_per_class": 10 if SMOKE else 100,
    "n_val_per_class": 10 if SMOKE else 500,
    "seeds": [0] if SMOKE else [0, 1, 2],
    # Budget rule from the optimizer pilot: 25 COBYLA evaluations per trainable parameter
    # (at that budget COBYLA matched the training loss of the converged L-BFGS-B).
    "evals_per_parameter": 2 if SMOKE else 25,
}

OUT_CSV = "results/ansatz_pilot_smoke.csv" if SMOKE else "results/ansatz_pilot.csv"
os.makedirs("results", exist_ok=True)
with open(OUT_CSV.replace(".csv", "_config.json"), "w") as fh:
    json.dump(PILOT, fh, indent=2)

# Resume: runs already in the CSV are skipped.
done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
def already_done(fmap, seed, ansatz, reps):
    return (not done.empty) and ((done.feature_map == fmap) & (done.seed == seed)
                                 & (done.ansatz == ansatz) & (done.reps == reps)).any()

for seed in PILOT["seeds"]:
    train, _, classes = load_scenario(PILOT["scenario"], PILOT["n_train_per_class"], seed)
    val, _ = load_validation(PILOT["scenario"], PILOT["n_val_per_class"], train["row_id"])

    for fmap in PILOT["feature_maps"]:
        Xtr, Xval, ytr, yval, info = to_qubits(train, val, reducer=PILOT["reducer"],
                                               n_qubits=PILOT["n_qubits"],
                                               angle_max=ANGLE_MAX[fmap], seed=seed)
        for ansatz, reps in PILOT["ansatz_candidates"]:
            if already_done(fmap, seed, ansatz, reps):
                print(f"skip {fmap} seed={seed} {ansatz} reps={reps}")
                continue
            circuit = build_ansatz(ansatz, PILOT["n_qubits"], reps)
            n_params = circuit.num_parameters
            maxiter = PILOT["evals_per_parameter"] * n_params
            # Seeded uniform start on [0, 2pi), as in the optimizer pilot.
            initial_point = np.random.default_rng(seed).uniform(0, 2 * np.pi, n_params)
            vqc = build_vqc(fmap, ansatz, PILOT["n_qubits"], reps, PILOT["n_classes"],
                            optimizer=COBYLA(maxiter=maxiter), initial_point=initial_point)
            t0 = time.time()
            vqc.fit(Xtr, ytr)
            seconds = time.time() - t0
            pred_val, pred_tr = vqc.predict(Xval), vqc.predict(Xtr)
            row = {"feature_map": fmap, "seed": seed, "ansatz": ansatz, "reps": reps,
                   "parameters": n_params,
                   "effective_parameters": effective_parameters(ansatz, circuit),
                   "maxiter": maxiter,
                   "val_macro_f1": round(f1_score(yval, pred_val, average="macro"), 4),
                   "val_accuracy": round(accuracy_score(yval, pred_val), 4),
                   # train F1 next to val F1 shows under- vs over-fitting
                   "train_macro_f1": round(f1_score(ytr, pred_tr, average="macro"), 4),
                   "final_train_loss": round(float(vqc.fit_result.fun), 4),
                   "loss_evaluations": int(vqc.fit_result.nfev),
                   "seconds": round(seconds, 1)}
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False,
                                       header=not os.path.exists(OUT_CSV))

# ---------------- Decision rule (fixed before running) ----------------
# One-standard-error rule (Breiman et al. 1984; Hastie et al., ESL sec. 7.10):
# 1. Best = configuration with the highest mean validation macro-F1.
# 2. Each other configuration is compared to the best RUN BY RUN (same feature map and
#    seed), which removes the RY-vs-ZZ gap and seed effects. It is "equivalent" to the
#    best if its mean paired gap is within one STANDARD ERROR of that gap (sd / sqrt(n)),
#    i.e. the data cannot tell them apart.
# 3. Among the best and its equivalents, choose the one with the FEWEST effective
#    parameters: cheaper training and less risk of overfitting.
res = pd.read_csv(OUT_CSV)
res["config"] = res["ansatz"] + "_r" + res["reps"].astype(str)
f1 = res.pivot_table(index=["feature_map", "seed"], columns="config", values="val_macro_f1")
params = res.groupby("config")["effective_parameters"].first()

best = f1.mean().idxmax()
rows = []
for cfg in f1.columns:
    diff = f1[best] - f1[cfg]                     # paired differences, one per (fmap, seed)
    se = diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    rows.append({"config": cfg, "val_f1_mean": f1[cfg].mean(),
                 "mean_gap_to_best": diff.mean(), "se_of_gap": se,
                 "equivalent_to_best": cfg == best or diff.mean() <= se,
                 "effective_parameters": params[cfg],
                 "train_f1_mean": res[res.config == cfg]["train_macro_f1"].mean(),
                 "seconds_mean": res[res.config == cfg]["seconds"].mean()})
summary = pd.DataFrame(rows).round(4).sort_values("val_f1_mean", ascending=False)
summary.to_csv(OUT_CSV.replace(".csv", "_summary.csv"), index=False)
chosen = summary[summary.equivalent_to_best].sort_values(
    ["effective_parameters", "val_f1_mean"], ascending=[True, False]).iloc[0]["config"]

print("\n", summary.to_string(index=False))
print("\nper feature map (val macro-F1):\n",
      res.pivot_table(index="config", columns="feature_map", values="val_macro_f1").round(4))
print(f"\nbest by mean F1: {best}   ->   chosen by the rule: {chosen}")