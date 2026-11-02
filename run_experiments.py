"""Main experiments: VQC (RY and ZZ encodings) and classical baselines on every scenario.

    python run_experiments.py --stage learning_curves --plan          # count runs, estimate hours
    python run_experiments.py --smoke                                  # 2-3 minute check
    python run_experiments.py --stage learning_curves --shard 0 --nshards 4   # run 1/4 of the tasks

Fixed model (decided by the pilots, see results/*_pilot_summary.csv):
  EfficientSU2, 2 layers, linear entanglement; readout on the last qubit(s); cross-entropy;
  COBYLA with 25 evaluations per trainable parameter; seeded uniform start on [0, 2pi).

A TASK is one data configuration: (scenario, training size, qubits, reducer, sampling, seed).
For each task the same training sample is used by every model, and every model is scored on
the same fixed TEST set (1000 per class) and on a VALIDATION set (500 per class, from the
training pool). Any later choice must be made on validation numbers, never on test numbers.

Outputs (one CSV per shard, resumable; together they are the results table):
  results/main/runs_<stage>_shard<i>of<k>.csv  one row per (task, model)
  results/main/curves/<task>_<model>.json       VQC loss after every COBYLA evaluation
Trained VQC weights are stored in each row, so the later extra test sets (larger, and
realistic class mix) can be evaluated without retraining.
"""
import argparse, glob, json, os, time, warnings
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from qiskit_machine_learning.optimizers import COBYLA

from qml_data import SCENARIOS, scenario_classes, load_scenario, load_validation, to_qubits
from feature_maps import ANGLE_MAX
from ansatz import build_ansatz
from vqc_model import build_vqc
from baselines import CANDIDATES, all_features, fit_tuned

warnings.filterwarnings("ignore", category=UserWarning)

MODEL = {"ansatz": "efficient_su2", "reps": 2, "evals_per_parameter": 25}
N_VAL_PER_CLASS = 500

# Each stage is a grid; tasks shared between stages (same key) are run only once.
STAGES = {
    # Core question: how does performance grow with training size, per attack scenario,
    # and does representative (k-means) sampling beat random sampling?
    "learning_curves": {
        "scenario": SCENARIOS, "n_train": [25, 50, 100, 200, 400, 800],
        "n_qubits": [4], "reducer": ["pca"], "sampling": ["random", "kmeans"],
        "seed": [0, 1, 2], "feature_maps": ["ry"],
    },
    # How many qubits/features, and PCA vs MI features, at one fixed training size.
    "qubits_features": {
        "scenario": SCENARIOS, "n_train": [200],
        "n_qubits": [2, 4, 6, 8], "reducer": ["pca", "mi"], "sampling": ["random"],
        "seed": [0, 1, 2], "feature_maps": ["ry"],
    },
        # First look: every scenario at one setting. Answers "which attacks can QML detect,
    # and how does it compare with classical?" quickly. Shared with both stages above.
    "first_look": {
        "scenario": SCENARIOS, "n_train": [200],
        "n_qubits": [4], "reducer": ["pca"], "sampling": ["random"],
        "seed": [0, 1, 2], "feature_maps": ["ry", "zz"],
    },
    # Tiny grid to check that everything runs end to end.
    "smoke": {
        "scenario": ["Recon", "multiclass"], "n_train": [10],
        "n_qubits": [4], "reducer": ["pca", "mi"], "sampling": ["random", "kmeans"],
        "seed": [0], "feature_maps": ["ry", "zz"],
    },
}
# ZZ is dropped from the big stages by the rule fixed before looking: in the first look,
# VQC-RY beat VQC-ZZ on VALIDATION macro-F1 by more than one standard error (paired by
# seed) in all 9 scenarios. RY vs ZZ is therefore answered by the first_look stage.

# Seconds per (sample x loss evaluation), EfficientSU2 r2, 4 qubits, measured in the
# first_look stage with 8 shards running in parallel. Used only for --plan estimates and
# for balancing shards; assumed to grow linearly with qubits.
SEC_PER_SAMPLE_EVAL_4Q = {"ry": 0.0019, "zz": 0.0027}


def tasks_for(stage):
    g = STAGES[stage]
    out = []
    for sc, n, q, red, samp, seed in product(g["scenario"], g["n_train"], g["n_qubits"],
                                             g["reducer"], g["sampling"], g["seed"]):
        if len(scenario_classes(sc)) == 8 and q < 3:
            continue   # 8 classes need 3 readout qubits (see vqc_model.readout)
        out.append({"scenario": sc, "n_train": n, "n_qubits": q, "reducer": red,
                    "sampling": samp, "seed": seed})
    return out, g["feature_maps"]


def task_id(t):
    return f"{t['scenario']}_n{t['n_train']}_q{t['n_qubits']}_{t['reducer']}_{t['sampling']}_s{t['seed']}"


def vqc_parameters(n_qubits):
    return build_ansatz(MODEL["ansatz"], n_qubits, MODEL["reps"]).num_parameters


def estimate_seconds(t, fmaps):
    n_classes = len(scenario_classes(t["scenario"]))
    evals = MODEL["evals_per_parameter"] * vqc_parameters(t["n_qubits"])
    samples = t["n_train"] * n_classes
    return sum(evals * samples * SEC_PER_SAMPLE_EVAL_4Q[f] * t["n_qubits"] / 4 for f in fmaps)

def assign_shards(tasks, fmaps, nshards):
    """Longest-task-first balancing: tasks sorted by estimated time (largest first) and each
    given to the shard with the least work so far, so all shards finish at about the same
    time. Deterministic (ties broken by task id), so every shard computes the same split."""
    load = [0.0] * nshards
    owner = {}
    for t in sorted(tasks, key=lambda t: (-estimate_seconds(t, fmaps), task_id(t))):
        k = min(range(nshards), key=lambda i: (load[i], i))
        owner[task_id(t)] = k
        load[k] += estimate_seconds(t, fmaps)
    return owner

def metrics(y_true, y_pred, n_classes):
    """Same keys for every scenario, so all rows share one column layout.
    macro_f1 / accuracy: overall quality. per_class_recall: share of each class found.
    Binary only (Benign=0 vs attack=1): detection_rate = attacks correctly flagged,
    false_alarm_rate = benign traffic wrongly flagged (the two numbers an IDS operator
    cares about); NaN for the 8-class scenario."""
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
         "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
         "detection_rate": round(float(cm[1, 1] / cm[1].sum()), 4) if n_classes == 2 else np.nan,
         "false_alarm_rate": round(float(cm[0, 1] / cm[0].sum()), 4) if n_classes == 2 else np.nan,
         "per_class_recall": json.dumps(recall_score(y_true, y_pred, average=None, labels=labels,
                                                     zero_division=0).round(4).tolist()),
         "confusion_matrix": json.dumps(cm.tolist())}
    return m


METRIC_KEYS = ["macro_f1", "accuracy", "detection_rate", "false_alarm_rate",
               "per_class_recall", "confusion_matrix"]
# Fixed column order for every row, whatever the model or scenario.
COLUMNS = (["task", "scenario", "n_train", "n_qubits", "reducer", "sampling", "seed", "model",
            "seconds"]
           + [f"test_{k}" for k in METRIC_KEYS] + [f"val_{k}" for k in METRIC_KEYS]
           + ["train_macro_f1", "final_train_loss", "loss_evaluations", "parameters",
              "features", "pca_explained_variance", "hyperparameter", "weights"])


def run_task(t, fmaps, done, out_csv, curves_dir):
    tid = task_id(t)
    classes = scenario_classes(t["scenario"])
    n_classes = len(classes)
    train, test, _ = load_scenario(t["scenario"], t["n_train"], t["seed"], strategy=t["sampling"])
    val, _ = load_validation(t["scenario"], N_VAL_PER_CLASS, train["row_id"])
    rows = []

    def record(model_name, pred_test, pred_val, seconds, extra=None):
        row = {"task": tid, **t, "model": model_name, "seconds": round(seconds, 1)}
        row.update({f"test_{k}": v for k, v in metrics(test["y"].to_numpy(), pred_test, n_classes).items()})
        row.update({f"val_{k}": v for k, v in metrics(val["y"].to_numpy(), pred_val, n_classes).items()})
        row.update(extra or {})
        rows.append(row)
        print({k: row[k] for k in ["task", "model", "test_macro_f1", "val_macro_f1", "seconds"]}, flush=True)

    # ---- VQC, one per feature map ----
    for fmap in fmaps:
        name = f"VQC-{fmap.upper()}"
        if (tid, name) in done:
            continue
        Xtr, Xte, ytr, yte, info = to_qubits(train, test, reducer=t["reducer"],
                                             n_qubits=t["n_qubits"], angle_max=ANGLE_MAX[fmap],
                                             seed=t["seed"])
        _, Xval, _, _, _ = to_qubits(train, val, reducer=t["reducer"], n_qubits=t["n_qubits"],
                                     angle_max=ANGLE_MAX[fmap], seed=t["seed"])
        n_params = vqc_parameters(t["n_qubits"])
        losses = []
        vqc = build_vqc(fmap, MODEL["ansatz"], t["n_qubits"], MODEL["reps"], n_classes,
                        optimizer=COBYLA(maxiter=MODEL["evals_per_parameter"] * n_params),
                        initial_point=np.random.default_rng(t["seed"]).uniform(0, 2 * np.pi, n_params),
                        callback=lambda w, loss: losses.append(round(float(loss), 5)))
        t0 = time.time()
        vqc.fit(Xtr, ytr)
        secs = time.time() - t0
        with open(f"{curves_dir}/{tid}_{name}.json", "w") as fh:
            json.dump(losses, fh)
        record(name, vqc.predict(Xte), vqc.predict(Xval), secs, {
            "train_macro_f1": round(f1_score(ytr, vqc.predict(Xtr), average="macro"), 4),
            "final_train_loss": round(float(vqc.fit_result.fun), 4),
            "loss_evaluations": int(vqc.fit_result.nfev), "parameters": n_params,
            "features": json.dumps(info.get("features", [])),
            "pca_explained_variance": round(info.get("explained_variance", np.nan), 4),
            "weights": json.dumps(np.round(vqc.weights, 6).tolist())})

    # ---- Classical on the SAME features the VQC sees (RY scaling; same information) ----
    Xtr, Xte, ytr, _, _ = to_qubits(train, test, reducer=t["reducer"], n_qubits=t["n_qubits"],
                                    angle_max=ANGLE_MAX["ry"], seed=t["seed"])
    _, Xval, _, _, _ = to_qubits(train, val, reducer=t["reducer"], n_qubits=t["n_qubits"],
                                 angle_max=ANGLE_MAX["ry"], seed=t["seed"])
    yval = val["y"].to_numpy()
    for name in CANDIDATES:
        if (tid, name) in done:
            continue
        t0 = time.time()
        model, hp = fit_tuned(name, Xtr, ytr, Xval, yval, t["seed"])
        record(name, model.predict(Xte), model.predict(Xval), time.time() - t0,
               {"hyperparameter": str(hp)})

    # ---- Reference: random forest on ALL 33 features (cost of the qubit bottleneck) ----
    if (tid, "RF-all33") not in done:
        Ftr, Fte, Fval = all_features(train, test, val)
        t0 = time.time()
        model, hp = fit_tuned("RF", Ftr, ytr, Fval, yval, t["seed"])
        record("RF-all33", model.predict(Fte), model.predict(Fval), time.time() - t0,
               {"hyperparameter": str(hp)})

    if rows:
        pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", index=False,
                                                          header=not os.path.exists(out_csv))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--plan", action="store_true", help="only count runs and estimate time")
    ap.add_argument("--smoke", action="store_true", help="shortcut for --stage smoke")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    stage = "smoke" if args.smoke else args.stage
    if stage is None:
        ap.error("give --stage (or --smoke)")
    if stage == "smoke":
        MODEL["evals_per_parameter"] = 2
        N_VAL_PER_CLASS = 20

    tasks, fmaps = tasks_for(stage)
    out_dir = "results/main_smoke" if stage == "smoke" else "results/main"
    curves_dir = f"{out_dir}/curves"
    os.makedirs(curves_dir, exist_ok=True)

    # Resume: (task, model) pairs already present in ANY shard file are skipped.
    done = set()
    for f in glob.glob(f"{out_dir}/runs_*.csv"):
        d = pd.read_csv(f, usecols=["task", "model"])
        done |= set(zip(d["task"], d["model"]))
    todo = [t for t in tasks if any((task_id(t), m) not in done
                                    for m in [f"VQC-{f.upper()}" for f in fmaps] + list(CANDIDATES) + ["RF-all33"])]

    if args.plan:
        hours = sum(estimate_seconds(t, fmaps) for t in todo) / 3600
        print(f"stage '{stage}': {len(tasks)} tasks ({len(tasks) - len(todo)} already done), "
              f"{len(todo) * len(fmaps)} VQC trainings to go")
        owner = assign_shards(tasks, fmaps, args.nshards)
        per_shard = [sum(estimate_seconds(t, fmaps) for t in todo if owner[task_id(t)] == k) / 3600
                     for k in range(args.nshards)]
        print(f"estimated VQC time: {hours:.1f} h on one process "
              f"-> {max(per_shard):.1f} h with {args.nshards} parallel shards "
              f"(busiest shard; others {min(per_shard):.1f}-{max(per_shard):.1f} h)")
        by_sc = pd.DataFrame([{**t, "hours": estimate_seconds(t, fmaps) / 3600} for t in todo])
        if len(by_sc):
            print(by_sc.groupby("scenario")["hours"].sum().round(1).to_string())
        raise SystemExit

    with open(f"{out_dir}/model_config.json", "w") as fh:
        json.dump({"model": MODEL, "stages": STAGES, "n_val_per_class": N_VAL_PER_CLASS}, fh, indent=2, default=str)
    #Balanced split of the FULL task list”.
    todo_ids = {task_id(t) for t in todo}
    owner = assign_shards(tasks, fmaps, args.nshards)
    mine = [t for t in tasks if owner[task_id(t)] == args.shard and task_id(t) in todo_ids]
    out_csv = f"{out_dir}/runs_{stage}_shard{args.shard}of{args.nshards}.csv"
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} tasks")
    for i, t in enumerate(mine, 1):
        print(f"--- [{i}/{len(mine)}] {task_id(t)}", flush=True)
        run_task(t, fmaps, done, out_csv, curves_dir)