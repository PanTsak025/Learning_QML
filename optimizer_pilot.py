"""Optimizer pilot: which optimizer trains the VQC best, measured on VALIDATION data.

    python optimizer_pilot.py --smoke   # 1-2 minute check that everything runs
    python optimizer_pilot.py           # the real pilot (can take a few hours; resumable)

Fixed for the pilot (only the optimizer varies):
  scenario benign_vs_attack (attack class = equal share of all 7 families, so it is
  representative of every attack type), 4 qubits, PCA, RealAmplitudes reps=2 (middle
  candidate), 100 training samples per class, both feature maps, 3 seeds.
The test set is never touched: every number below comes from a validation set drawn from
the TRAINING pool and disjoint from the training sample.
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit_machine_learning.utils import algorithm_globals

from qml_data import load_scenario, load_validation, to_qubits
from feature_maps import ANGLE_MAX
from ansatz import build_ansatz
from vqc_model import build_vqc

warnings.filterwarnings("ignore", category=UserWarning)   # COBYLA prints a harmless MAXFUN note

SMOKE = "--smoke" in sys.argv
PILOT = {
    "scenario": "benign_vs_attack", "n_classes": 2,
    "feature_maps": ["ry", "zz"],
    "ansatz": "real_amplitudes", "reps": 2, "n_qubits": 4, "reducer": "pca",
    "n_train_per_class": 10 if SMOKE else 100,
    "n_val_per_class": 10 if SMOKE else 500,
    "seeds": [0] if SMOKE else [0, 1, 2],
}
# Iteration caps. Each optimizer gets a cap generous enough to converge on 12 parameters;
# actual cost (loss evaluations, seconds) is recorded, so the comparison includes price.
#   COBYLA   : gradient-free, 1 loss evaluation per iteration (Powell 1994).
#   SPSA     : stochastic gradient estimate from 2 evaluations per iteration, whatever the
#              number of parameters; designed for noisy (shot-based) losses (Spall 1992).
#   L-BFGS-B : exact gradients by parameter shift (2 extra circuit runs per parameter),
#              quasi-Newton steps; few iterations but each one is expensive.
MAXITER = {"COBYLA": 300, "SPSA": 150, "L-BFGS-B": 100}
if SMOKE:
    MAXITER = {"COBYLA": 30, "SPSA": 15, "L-BFGS-B": 5}
OPTIMIZERS = {
    "COBYLA":   lambda: COBYLA(maxiter=MAXITER["COBYLA"]),
    "SPSA":     lambda: SPSA(maxiter=MAXITER["SPSA"]),
    "L-BFGS-B": lambda: L_BFGS_B(maxiter=MAXITER["L-BFGS-B"]),
}

OUT_CSV = "results/optimizer_pilot_smoke.csv" if SMOKE else "results/optimizer_pilot.csv"
os.makedirs("results", exist_ok=True)
with open(OUT_CSV.replace(".csv", "_config.json"), "w") as fh:
    json.dump({"pilot": PILOT, "maxiter": MAXITER}, fh, indent=2)

# Resume: runs already in the CSV are skipped, so a long pilot can be stopped and restarted.
done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
def already_done(fmap, seed, opt):
    return (not done.empty) and ((done.feature_map == fmap) & (done.seed == seed)
                                 & (done.optimizer == opt)).any()

n_weights = build_ansatz(PILOT["ansatz"], PILOT["n_qubits"], PILOT["reps"]).num_parameters

for seed in PILOT["seeds"]:
    train, _, classes = load_scenario(PILOT["scenario"], PILOT["n_train_per_class"], seed)
    val, _ = load_validation(PILOT["scenario"], PILOT["n_val_per_class"], train["row_id"])

    # Same starting weights for every optimizer and feature map within a seed: uniform on
    # [0, 2pi), the full period of the rotation gates, so no starting point is favoured.
    # Paired starts mean differences come from the optimizer, not from luck of the start.
    initial_point = np.random.default_rng(seed).uniform(0, 2 * np.pi, n_weights)

    for fmap in PILOT["feature_maps"]:
        # to_qubits fits every transform on the training sample only; validation is only transformed.
        Xtr, Xval, ytr, yval, info = to_qubits(train, val, reducer=PILOT["reducer"],
                                               n_qubits=PILOT["n_qubits"],
                                               angle_max=ANGLE_MAX[fmap], seed=seed)
        for opt_name, make_opt in OPTIMIZERS.items():
            if already_done(fmap, seed, opt_name):
                print(f"skip {fmap} seed={seed} {opt_name} (already in {OUT_CSV})")
                continue
            algorithm_globals.random_seed = seed     # makes SPSA's random directions reproducible
            vqc = build_vqc(fmap, PILOT["ansatz"], PILOT["n_qubits"], PILOT["reps"],
                            PILOT["n_classes"], optimizer=make_opt(),
                            initial_point=initial_point)
            t0 = time.time()
            vqc.fit(Xtr, ytr)
            seconds = time.time() - t0
            pred = vqc.predict(Xval)
            res = vqc.fit_result
            row = {"feature_map": fmap, "seed": seed, "optimizer": opt_name,
                   "val_macro_f1": round(f1_score(yval, pred, average="macro"), 4),
                   "val_accuracy": round(accuracy_score(yval, pred), 4),
                   "final_train_loss": round(float(res.fun), 4),
                   "loss_evaluations": res.nfev,
                   "gradient_evaluations": res.njev,
                   "iterations": res.nit,
                   "seconds": round(seconds, 1)}
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False,
                                       header=not os.path.exists(OUT_CSV))

# Summary: mean and std over (feature map x seed) per optimizer.
res = pd.read_csv(OUT_CSV)
summary = res.groupby("optimizer").agg(
    f1_mean=("val_macro_f1", "mean"), f1_std=("val_macro_f1", "std"),
    loss_mean=("final_train_loss", "mean"), seconds_mean=("seconds", "mean"),
    evaluations_mean=("loss_evaluations", "mean")).round(4).sort_values("f1_mean", ascending=False)
summary.to_csv(OUT_CSV.replace(".csv", "_summary.csv"))
print("\n", summary)
print("\nper feature map:\n", res.pivot_table(index="optimizer", columns="feature_map",
                                               values="val_macro_f1", aggfunc="mean").round(4))