"""Extra evaluations of the trained learning-curve models. NO VQC is retrained: every VQC is
rebuilt from its saved weights; classical models are refitted with their saved
hyperparameter on the identical training sample (deterministic, a few seconds each).

    python evaluate_extra.py --limit 2            # quick check on 2 tasks (results/main/eval_extra_check.csv)
    python evaluate_extra.py --shard 0 --nshards 8 # full run, one of 8 parallel shards
    python evaluate_extra.py --summary            # tables from the finished evaluation

Evaluation sets (all drawn from the TEST pool, so disjoint from every training and
validation row, and excluding the 1,000-per-class fixed test set used so far):
  fixed_test   : the original fixed test set. Used as a REPRODUCTION CHECK: rebuilt models
                 must give the same test macro-F1 as stored during training (repro_diff ~ 0).
  extended     : larger balanced test set, up to 10,000 per class, limited by the smallest
                 class in the scenario (BruteForce has 1,502 unused test rows, Web 3,703).
                 Purpose: tighter confidence intervals.
  realistic    : 20,000 rows with the NATURAL class mix of the cleaned test pool (e.g. Benign
                 vs all attacks: ~5% benign). Purpose: real-world false alarms and precision.
  shots1024    : fixed test set, VQC measured with 1,024 shots instead of exact probabilities.
  noisy        : fixed test set, VQC on a generic noisy simulator (Qiskit Aer): depolarizing
                 error 0.1% per 1-qubit gate, 1% per CNOT, 2% readout error, 1,024 shots.
                 Order of magnitude of current superconducting devices; not a specific device.
"""
import argparse, glob, json, os, warnings
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import QMLSampler
from sklearn.neural_network import MLPClassifier

from qml_data import KEPT, ATTACKS, scenario_classes, load_scenario, to_qubits, _split_evenly
from feature_maps import build_feature_map, ANGLE_MAX
from ansatz import build_ansatz
from vqc_model import readout
from baselines import CANDIDATES, all_features
from run_experiments import tasks_for, task_id, metrics, MODEL

warnings.filterwarnings("ignore")

EVAL = {"seed": 2026, "extended_per_class": 10000, "realistic_total": 20000, "shots": 1024,
        "noise": {"p_1q": 0.001, "p_cx": 0.01, "p_readout": 0.02}}
OUT, PREP = "results/main", "prepared"
FAMILIES = ["Benign"] + ATTACKS
MODELS = ["VQC-RY", "LogReg", "SVM-RBF", "RF", "MLP-matched", "RF-all33"]
COLUMNS = ["task", "scenario", "n_train", "n_qubits", "reducer", "sampling", "seed", "model",
           "eval_set", "n_samples", "macro_f1", "accuracy", "detection_rate", "false_alarm_rate",
           "precision", "per_class_recall", "confusion_matrix", "repro_diff"]


# ---------------------------------------------------------------- evaluation pools
def build_eval_pools():
    """Per family: test-pool rows NOT in the fixed test set, in a seeded random order that is
    identical on every machine (sorted by content hash first, as in build_pools.py)."""
    os.makedirs(f"{PREP}/eval", exist_ok=True)
    keep = max(EVAL["extended_per_class"], EVAL["realistic_total"])
    for fam in FAMILIES:
        path = f"{PREP}/eval/test_{fam}.parquet"
        if os.path.exists(path):
            continue
        d = pd.read_parquet(f"{PREP}/test", columns=KEPT + ["Label"], filters=[("family", "==", fam)])
        d["row_id"] = pd.util.hash_pandas_object(d[KEPT + ["Label"]], index=False).to_numpy()
        fixed = set(pd.read_parquet(f"{PREP}/pools/test_{fam}.parquet")["row_id"])
        d = d[~d["row_id"].isin(fixed)].sort_values("row_id", kind="stable")
        d = d.sample(frac=1, random_state=EVAL["seed"]).head(keep).reset_index(drop=True)
        d.to_parquet(path, index=False)
        print(f"eval pool {fam}: {len(d):,} rows")


_cache = {}
def _ev(fam):
    if fam not in _cache:
        _cache[fam] = pd.read_parquet(f"{PREP}/eval/test_{fam}.parquet")
    return _cache[fam]


def class_families(cls):
    return ATTACKS if cls == "Attack" else [cls]


def extended_set(scenario):
    classes = scenario_classes(scenario)
    avail = []
    for cls in classes:
        fams = class_families(cls)
        avail.append(min(len(_ev(f)) for f in fams) * len(fams))
    n = min([EVAL["extended_per_class"]] + avail)        # balanced, limited by the smallest class
    parts = []
    for y, cls in enumerate(classes):
        fams = class_families(cls)
        for f, k in zip(fams, _split_evenly(n, len(fams))):
            parts.append(_ev(f).head(k).assign(y=y))
    return pd.concat(parts, ignore_index=True)


def realistic_set(scenario):
    sizes = pd.read_csv(f"{PREP}/cleaning_report_by_family.csv", index_col=0)["test"]
    classes = scenario_classes(scenario)
    fams = [(y, f) for y, cls in enumerate(classes) for f in class_families(cls)]
    total = sum(sizes[f] for _, f in fams)
    parts = []
    for y, f in fams:
        k = max(1, int(round(EVAL["realistic_total"] * sizes[f] / total)))   # natural share, >= 1 row
        parts.append(_ev(f).head(min(k, len(_ev(f)))).assign(y=y))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- models
def vqc_qnn(n_qubits, n_classes, sampler):
    """Same circuit and parameter order as the VQC used in training."""
    fm, an = build_feature_map("ry", n_qubits), build_ansatz(MODEL["ansatz"], n_qubits, MODEL["reps"])
    c = QuantumCircuit(n_qubits)
    c.compose(fm, inplace=True)
    c.compose(an, inplace=True)
    interpret, shape = readout(n_qubits, n_classes)
    return SamplerQNN(circuit=c, input_params=fm.parameters, weight_params=an.parameters,
                      interpret=interpret, output_shape=shape, sampler=sampler)


def noisy_sampler(seed):
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    except ImportError:
        return None
    nm, p = NoiseModel(), EVAL["noise"]
    nm.add_all_qubit_quantum_error(depolarizing_error(p["p_1q"], 1), ["ry", "rz", "rx", "h", "p", "u", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p["p_cx"], 2), ["cx"])
    r = p["p_readout"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return AerSampler(default_shots=EVAL["shots"], seed=seed, options={"backend_options": {"noise_model": nm}})


def refit_classical(name, hp, Xtr, ytr, seed):
    if name == "MLP-matched":
        h = int(hp.split("h=")[1].split(",")[0])
        a = float(hp.split("alpha=")[1])
        return MLPClassifier(hidden_layer_sizes=(h,), activation="tanh", solver="lbfgs", alpha=a,
                             max_iter=2000, random_state=seed).fit(Xtr, ytr)
    base = "RF" if name == "RF-all33" else name
    p = None if hp in ("None", "nan") else (int(float(hp)) if base == "RF" else float(hp))
    return CANDIDATES[base][0](p, seed).fit(Xtr, ytr)


def score(y, pred, n_classes):
    m = metrics(y, pred, n_classes)
    cm = np.array(json.loads(m["confusion_matrix"]))
    m["precision"] = round(float(cm[1, 1] / cm[:, 1].sum()), 4) if n_classes == 2 and cm[:, 1].sum() else np.nan
    return m


# ---------------------------------------------------------------- one task
def evaluate_task(t, stored, done, out_csv):
    tid = task_id(t)
    n_classes = len(scenario_classes(t["scenario"]))
    train, fixed, _ = load_scenario(t["scenario"], t["n_train"], t["seed"], strategy=t["sampling"])
    sets = {"fixed_test": fixed, "extended": extended_set(t["scenario"]),
            "realistic": realistic_set(t["scenario"])}
    big = pd.concat([d.assign(_set=k) for k, d in sets.items()], ignore_index=True)
    masks = {k: (big["_set"] == k).to_numpy() for k in sets}
    y = big["y"].to_numpy()
    Xtr, Xbig, ytr, _, _ = to_qubits(train, big, reducer=t["reducer"], n_qubits=t["n_qubits"],
                                     angle_max=ANGLE_MAX["ry"], seed=t["seed"])
    rows = []

    def add(model, eval_set, yy, pred, repro=np.nan):
        rows.append({"task": tid, **t, "model": model, "eval_set": eval_set, "n_samples": len(yy),
                     **score(yy, pred, n_classes), "repro_diff": repro})

    def stored_f1(model):
        r = stored[(stored.task == tid) & (stored.model == model)]
        return float(r["test_macro_f1"].iloc[0]) if len(r) else np.nan

    # ---- VQC from saved weights
    r = stored[(stored.task == tid) & (stored.model == "VQC-RY")]
    if len(r) and (tid, "VQC-RY", "noisy") not in done:
        w = np.array(json.loads(r["weights"].iloc[0]))
        pred = vqc_qnn(t["n_qubits"], n_classes, QMLSampler()).forward(Xbig, w).argmax(1)
        for k, m in masks.items():
            f1 = score(y[m], pred[m], n_classes)["macro_f1"]
            add("VQC-RY", k, y[m], pred[m], round(f1 - stored_f1("VQC-RY"), 4) if k == "fixed_test" else np.nan)
        Xf, yf = Xbig[masks["fixed_test"]], y[masks["fixed_test"]]
        shots = QMLSampler(shots=EVAL["shots"], seed=t["seed"])
        add("VQC-RY", "shots1024", yf, vqc_qnn(t["n_qubits"], n_classes, shots).forward(Xf, w).argmax(1))
        ns = noisy_sampler(t["seed"])
        if ns is not None:
            add("VQC-RY", "noisy", yf, vqc_qnn(t["n_qubits"], n_classes, ns).forward(Xf, w).argmax(1))

    # ---- classical, refitted with the stored hyperparameter on the same training sample
    Ftr = Fbig = None
    for name in MODELS[1:]:
        r = stored[(stored.task == tid) & (stored.model == name)]
        if not len(r) or (tid, name, "realistic") in done:
            continue
        if name == "RF-all33":
            if Ftr is None:
                Ftr, Fbig = all_features(train, big)
            model = refit_classical(name, str(r["hyperparameter"].iloc[0]), Ftr, ytr, t["seed"])
            pred = model.predict(Fbig)
        else:
            model = refit_classical(name, str(r["hyperparameter"].iloc[0]), Xtr, ytr, t["seed"])
            pred = model.predict(Xbig)
        for k, m in masks.items():
            f1 = score(y[m], pred[m], n_classes)["macro_f1"]
            add(name, k, y[m], pred[m], round(f1 - stored_f1(name), 4) if k == "fixed_test" else np.nan)

    if rows:
        pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", index=False,
                                                          header=not os.path.exists(out_csv))
        print({"task": tid, "rows": len(rows),
               "max_repro_diff": float(np.nanmax(np.abs([r["repro_diff"] for r in rows])))}, flush=True)


# ---------------------------------------------------------------- summary
def summary():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/eval_extra_shard*.csv")], ignore_index=True)
    d = d.drop_duplicates(["task", "model", "eval_set"])
    pd.set_option("display.width", 250)
    print(f"{len(d)} evaluation rows, {d.task.nunique()} tasks")
    rep = d[d.eval_set == "fixed_test"].groupby("model")["repro_diff"].agg(lambda s: s.abs().max())
    print("\n1) Reproduction check: largest |rebuilt - stored| test macro-F1 per model "
          "(should be ~0)\n", rep.round(4).to_string())
    size = int(d.n_train.max())        # the largest training size (800 per class in the study)
    big = d[(d.n_train == size) & (d.sampling == "random")]
    order = list(dict.fromkeys(t["scenario"] for t in tasks_for("learning_curves")[0]))
    for es, title in [("extended", f"2) EXTENDED balanced test set ({size}/class training, random): macro-F1"),
                      ("realistic", f"3) REALISTIC class mix ({size}/class training, random): macro-F1")]:
        g = big[big.eval_set == es].groupby(["scenario", "model"])["macro_f1"]
        t = (g.mean().round(3).astype(str) + "±" + (g.std() / np.sqrt(g.count())).round(3).astype(str)).unstack()
        print(f"\n{title}\n", t.reindex(index=order, columns=MODELS).to_string())
    rb = big[(big.eval_set == "realistic") & (big.scenario != "multiclass")]
    for col, title in [("false_alarm_rate", "4a) REALISTIC mix: false-alarm rate (benign wrongly flagged)"),
                       ("precision", "4b) REALISTIC mix: precision (share of alerts that are real attacks)")]:
        t = rb.groupby(["scenario", "model"])[col].mean().round(3).unstack()
        print(f"\n{title}\n", t.reindex(index=[s for s in order if s != "multiclass"], columns=MODELS).to_string())
    q = d[d.model == "VQC-RY"].pivot_table(index=["scenario", "n_train", "sampling", "seed"],
                                          columns="eval_set", values="macro_f1")
    if "noisy" in q:
        drop = pd.DataFrame({"shots1024 - exact": (q["shots1024"] - q["fixed_test"]),
                             "noisy - exact": (q["noisy"] - q["fixed_test"])}).groupby("scenario").mean().round(4)
        print("\n5) VQC under shot noise and hardware noise: mean change in fixed-test macro-F1 "
              "(all sizes, both samplings, all seeds)\n", drop.reindex(order).to_string())
    d.to_csv(f"{OUT}/eval_extra_all.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="quick check on the first N tasks")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        raise SystemExit
    build_eval_pools()
    stored = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
    stored = stored[stored.model.isin(MODELS)].drop_duplicates(["task", "model"])
    tasks = tasks_for("learning_curves")[0]
    if args.limit:
        tasks = tasks[:args.limit]
        out_csv = f"{OUT}/eval_extra_check.csv"
        if os.path.exists(out_csv):
            os.remove(out_csv)          # a fresh check each time
        mine = tasks
    else:
        out_csv = f"{OUT}/eval_extra_shard{args.shard}of{args.nshards}.csv"
        mine = tasks[args.shard::args.nshards]
    # Resume: only the shard files count (the --limit check file is separate and not reused).
    done = set()
    for f in ([] if args.limit else glob.glob(f"{OUT}/eval_extra_shard*.csv")):
        e = pd.read_csv(f, usecols=["task", "model", "eval_set"])
        done |= set(zip(e.task, e.model, e.eval_set))
    with open(f"{OUT}/eval_extra_config.json", "w") as fh:
        json.dump(EVAL, fh, indent=2)
    print(f"{len(mine)} tasks to evaluate -> {out_csv}")
    for i, t in enumerate(mine, 1):
        print(f"--- [{i}/{len(mine)}] {task_id(t)}", flush=True)
        evaluate_task(t, stored, done, out_csv)