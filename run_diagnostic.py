"""Diagnostic stage: WHY does the VQC underfit? Everything below was FIXED BEFORE the stage ran.

    python run_diagnostic.py --check                       # self-checks (about 1 minute)
    python run_diagnostic.py --smoke                       # tiny end-to-end run (a few minutes)
    python run_diagnostic.py --plan --nshards 8            # count jobs, estimate hours
    python run_diagnostic.py --shard i --nshards 8         # i = 0..7, run in parallel
    python analyze_diagnostic.py                           # decision rule and tables

This file only ADDS a stage. It changes no existing file and no existing result: it imports
the data pipeline, circuit builders, metrics and baselines unchanged, and writes only to
results/diagnostic/ (file names diag_*.csv), which no other script reads.

Why this stage exists
  In the main study the VQC underfits: training F1 is within 0.01 of test F1, its training
  cross-entropy falls only to 0.68-0.95 of a uniform guess outside the flood scenarios, and
  classical models reach higher scores on the SAME features. So the inputs carry more
  information than the trained circuit extracts. Three parts of the trained circuit could be
  responsible: the optimiser, the single-qubit readout, or the circuit's capacity. Each arm
  below changes exactly ONE of them, starting from the main-study VQC, so the arm that
  improves the training fit identifies the cause.

Common setting (the "first look" setting, run for every scenario in the main study)
  4 qubits, PCA, 200 training rows per class, random sampling, seeds 0, 1, 2; the same
  training rows, validation rows (500 per class) and test set as the main study.
  Scenarios: the five non-flood binary scenarios (primary) and 8-class (secondary).
  The flood scenarios are excluded for the same reason as in the paper's tests: one feature
  separates them, so they cannot show underfitting.

Arms (each one change from A0)
  A0 baseline   The main-study VQC, built by vqc_model.build_vqc exactly as in
                run_experiments.py: RY encoding, EfficientSU2 2 reps linear, last-qubit
                readout (8-class: last 3 qubits), cross-entropy in bits, COBYLA with
                25 evaluations per parameter, start uniform on [0, 2pi) from the seed.
                Re-run here so all arms share one machine and library state; it must
                reproduce the stored main-study result (checked in analyze_diagnostic.py).
  A1 restarts   OPTIMISER. Same model and budget, 4 more random starts
                (rng([seed, k]), k = 1..4). The arm's result is the start with the lowest
                final TRAINING loss among A0's start and these 4 (choice on training data only).
  A2 readout    READOUT. Same circuit, but every qubit is read: z_i = <Z_i> (i = 1..n), and a
                linear head maps z to class probabilities (binary: sigmoid(w.z + b),
                n + 1 parameters; 8-class: softmax(Wz + b), 8n + 8 parameters). Circuit and head
                are trained together with the same loss, optimiser and budget rule
                (25 evaluations per trainable parameter). Circuit start = A0's start, head
                start = 0 (so the first prediction is the uniform guess).
  A3 deeper     CAPACITY (more layers). EfficientSU2 with 4 reps instead of 2
                (40 parameters instead of 24 at 4 qubits), everything else as A0.
  A4 reupload   CAPACITY (data re-uploading, Perez-Salinas et al. 2020). The RY encoding is
                repeated before each of the 3 rotation layers of EfficientSU2 2 reps, with the
                SAME 24 trainable parameters as A0 and the same readout and training. Without
                the repeats the circuit equals A0's exactly (checked by --check).
  R2 head only  CONTROL for A2. The same linear head, loss, optimiser and budget rule, on the
                <Z_i> of the encoding alone (no ansatz), i.e. on cos(x_i). A trained head can
                raise scores without any help from the circuit (on cos(x) it is close to
                logistic regression), so A2 counts as evidence about the readout only if it
                also beats R2.
  A5 kernel     REFERENCE, no trained circuit. SVM on the fidelity kernel of the same RY
                encoding, k(x, x') = |<phi(x)|phi(x')>|^2, C in {0.1, 1, 10} chosen on
                validation macro-F1 (as SVM-RBF in baselines.py). Any VQC that uses this
                encoding once, with any ansatz and any single measurement, is a linear model in
                this kernel's feature space (Schuld 2021), so A5 shows how far the encoding
                itself can go. A4 is outside that family.

Recorded per (task, arm): test and validation metrics (run_experiments.metrics, unchanged),
training macro-F1, final training loss in bits and relative to a uniform guess (log2 C bits),
loss evaluations, trainable parameters, CPU time, trained weights, and the loss trace.
"""
import argparse, glob, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler

from qml_data import scenario_classes, load_scenario, load_validation, to_qubits
from feature_maps import build_feature_map, ANGLE_MAX
from ansatz import build_ansatz, ANSATZ_ENTANGLEMENT
from vqc_model import build_vqc, readout
from run_experiments import MODEL, N_VAL_PER_CLASS, metrics, METRIC_KEYS, SEC_PER_SAMPLE_EVAL_4Q

warnings.filterwarnings("ignore", category=UserWarning)

DIAG = {
    "scenarios": ["Recon", "Spoofing", "BruteForce", "Web", "benign_vs_attack", "multiclass"],
    "primary_scenarios": ["Recon", "Spoofing", "BruteForce", "Web", "benign_vs_attack"],
    "n_train": 200, "n_qubits": 4, "reducer": "pca", "sampling": "random", "seeds": [0, 1, 2],
    "extra_starts": [1, 2, 3, 4],          # A1: four starts besides A0's
    "deeper_reps": 4,                      # A3
    "kernel_C": [0.1, 1, 10],              # A5, same grid as SVM-RBF
}
OUT = "results/diagnostic"
COLUMNS = (["task", "scenario", "n_train", "n_qubits", "reducer", "sampling", "seed", "arm", "start",
            "seconds", "cpu_seconds"]
           + [f"test_{k}" for k in METRIC_KEYS] + [f"val_{k}" for k in METRIC_KEYS]
           + ["train_macro_f1", "final_train_loss", "train_loss_bits", "uniform_loss_bits",
              "loss_ratio", "loss_evaluations", "parameters", "hyperparameter", "weights"])


def use_smoke_settings():
    """Tiny grid for an end-to-end check (also used by analyze_diagnostic.py --smoke)."""
    global MODEL, N_VAL_PER_CLASS, OUT
    DIAG.update(scenarios=["Recon", "multiclass"], primary_scenarios=["Recon"], seeds=[0],
                n_train=10, extra_starts=[1])
    MODEL = dict(MODEL, evals_per_parameter=2)
    N_VAL_PER_CLASS = 20
    OUT = "results/diagnostic_smoke"


# ------------------------------------------------------------------ circuits and models
def reupload_circuit(n_qubits, reps, reupload=True):
    """EfficientSU2 (RY+RZ layers, linear CNOT chain) with the RY encoding placed before
    EVERY rotation layer (reupload=True) or only before the first (reupload=False, which is
    exactly the main-study circuit: RY encoding followed by efficient_su2). Same parameter
    names as efficient_su2 ('θ'), so the parameter order and count are identical."""
    x = ParameterVector("x", n_qubits)
    th = ParameterVector("θ", 2 * n_qubits * (reps + 1))
    qc = QuantumCircuit(n_qubits)
    k = 0
    for layer in range(reps + 1):
        if layer == 0 or reupload:
            for q in range(n_qubits):
                qc.ry(x[q], q)
        for q in range(n_qubits):
            qc.ry(th[k + q], q)
        for q in range(n_qubits):
            qc.rz(th[k + n_qubits + q], q)
        k += 2 * n_qubits
        if layer < reps:
            for q in range(n_qubits - 1):   # linear entanglement: CNOT(q, q+1)
                qc.cx(q, q + 1)
    return qc, x, th


def classifier_from_circuit(qc, x, th, n_qubits, n_classes, optimizer, initial_point, callback):
    """Same construction as qiskit's VQC (SamplerQNN + NeuralNetworkClassifier, one-hot targets,
    cross-entropy, exact QMLSampler, our readout), but for a circuit whose inputs and weights
    are interleaved, which VQC's feature_map + ansatz interface cannot express."""
    interpret, output_shape = readout(n_qubits, n_classes)
    qnn = SamplerQNN(sampler=QMLSampler(), circuit=qc, input_params=list(x), weight_params=list(th),
                     interpret=interpret, output_shape=output_shape, input_gradients=False)
    return NeuralNetworkClassifier(qnn, loss="cross_entropy", one_hot=True, optimizer=optimizer,
                                   initial_point=initial_point, callback=callback)


class ReadoutHeadModel:
    """A2: the main-study circuit, all qubits read as <Z_i>, a linear head on top; circuit and
    head trained jointly on mean cross-entropy in bits (log2, clipped at 1e-10 as in qiskit's
    CrossEntropyLoss) with qiskit's COBYLA."""

    def __init__(self, n_qubits, n_classes, reps):
        self.n, self.C = n_qubits, n_classes
        fmap = build_feature_map("ry", n_qubits)
        ansatz = build_ansatz(MODEL["ansatz"], n_qubits, reps)
        qc = QuantumCircuit(n_qubits)
        qc.compose(fmap, inplace=True)
        qc.compose(ansatz, inplace=True)
        self.qnn = SamplerQNN(sampler=QMLSampler(), circuit=qc, input_params=fmap.parameters,
                              weight_params=ansatz.parameters, interpret=lambda b: b,
                              output_shape=2 ** n_qubits, input_gradients=False)
        self.n_circuit = ansatz.num_parameters
        self.n_head = head_parameters(n_qubits, n_classes)
        self.num_parameters = self.n_circuit + self.n_head
        bits = (np.arange(2 ** n_qubits)[:, None] >> np.arange(n_qubits)[None, :]) & 1
        self.z_sign = 1 - 2 * bits                          # <Z_q> = sum_b P(b) (1 - 2 b_q)

    def expectations(self, X, theta):
        return self.qnn.forward(X, theta) @ self.z_sign      # shape (samples, n_qubits)

    def proba(self, X, w):
        return head_proba(self.expectations(X, w[:self.n_circuit]), w[self.n_circuit:], self.C)

    def fit(self, X, y, maxiter, initial_point, callback):
        def loss(w):
            value = mean_ce_bits(self.proba(X, w), y)
            callback(w, value)
            return value
        self.result = COBYLA(maxiter=maxiter).minimize(loss, initial_point)
        self.weights = np.asarray(self.result.x)
        return self

    def predict(self, X):
        return self.proba(X, self.weights).argmax(axis=1)


def head_parameters(n_qubits, n_classes):
    return (n_qubits + 1) if n_classes == 2 else (n_classes * n_qubits + n_classes)


def head_proba(z, h, n_classes):
    """Linear head on z (samples x n): binary sigmoid(w.z + b); C classes softmax(Wz + b)."""
    n = z.shape[1]
    if n_classes == 2:
        p1 = 1 / (1 + np.exp(-np.clip(z @ h[:n] + h[n], -500, 500)))
        return np.column_stack([1 - p1, p1])
    logits = z @ h[:n_classes * n].reshape(n, n_classes) + h[n_classes * n:]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def mean_ce_bits(probs, y):
    return float(np.mean(-np.log2(np.clip(probs[np.arange(len(y)), y], 1e-10, None))))


def ry_fidelity_kernel(A, B):
    """Closed form of the fidelity kernel of the RY product encoding (used only to CHECK
    qiskit's FidelityStatevectorKernel in --check): prod_i cos^2((a_i - b_i) / 2)."""
    return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)


# ------------------------------------------------------------------ jobs
def task_id(scenario, seed):
    d = DIAG
    return f"{scenario}_n{d['n_train']}_q{d['n_qubits']}_{d['reducer']}_{d['sampling']}_s{seed}"


def jobs():
    """One job = (scenario, seed, arm, start). A1's starts are separate jobs so shards balance."""
    out = []
    for sc in DIAG["scenarios"]:
        for seed in DIAG["seeds"]:
            out.append((sc, seed, "A0_baseline", 0))
            out += [(sc, seed, "A1_restart", k) for k in DIAG["extra_starts"]]
            out += [(sc, seed, a, 0) for a in ["A2_readout", "R2_head_only", "A3_deeper",
                                               "A4_reupload", "A5_kernel"]]
    return out


def job_key(j):
    return f"{task_id(j[0], j[1])}|{j[2]}|{j[3]}"


def n_trainable(arm, n_qubits, n_classes):
    if arm == "R2_head_only":
        return head_parameters(n_qubits, n_classes)
    reps = DIAG["deeper_reps"] if arm == "A3_deeper" else MODEL["reps"]
    p = build_ansatz(MODEL["ansatz"], n_qubits, reps).num_parameters
    if arm == "A2_readout":
        p += head_parameters(n_qubits, n_classes)
    return 0 if arm == "A5_kernel" else p


def estimate_seconds(j):
    """Rough planning number: loss evaluations x training samples x the main study's measured
    cost per sample-evaluation at 4 qubits (8 shards in parallel), times a depth factor."""
    sc, _, arm, _ = j
    C = len(scenario_classes(sc))
    if arm in ("A5_kernel", "R2_head_only"):
        return 30.0
    depth = {"A3_deeper": 1.7, "A4_reupload": 1.1}.get(arm, 1.0)
    evals = MODEL["evals_per_parameter"] * n_trainable(arm, DIAG["n_qubits"], C)
    return evals * DIAG["n_train"] * C * SEC_PER_SAMPLE_EVAL_4Q["ry"] * depth


def assign_shards(all_jobs, nshards):
    """Longest-job-first balancing; deterministic, so every shard computes the same split."""
    load, owner = [0.0] * nshards, {}
    for j in sorted(all_jobs, key=lambda j: (-estimate_seconds(j), job_key(j))):
        k = min(range(nshards), key=lambda i: (load[i], i))
        owner[job_key(j)] = k
        load[k] += estimate_seconds(j)
    return owner, load


# ------------------------------------------------------------------ one job
def run_job(j, out_csv, curves_dir):
    sc, seed, arm, start = j
    d = DIAG
    n, C = d["n_qubits"], len(scenario_classes(sc))
    train, test, _ = load_scenario(sc, d["n_train"], seed, strategy=d["sampling"])
    val, _ = load_validation(sc, N_VAL_PER_CLASS, train["row_id"])
    # Same feature pipeline and RY angle range as the main study (fitted on training rows only).
    Xtr, Xte, ytr, yte, _ = to_qubits(train, test, reducer=d["reducer"], n_qubits=n,
                                      angle_max=ANGLE_MAX["ry"], seed=seed)
    _, Xval, _, _, _ = to_qubits(train, val, reducer=d["reducer"], n_qubits=n,
                                 angle_max=ANGLE_MAX["ry"], seed=seed)
    yval = val["y"].to_numpy()
    losses = []
    callback = lambda w, loss: losses.append(round(float(loss), 5))
    extra = {"hyperparameter": ""}
    t0, c0 = time.time(), time.process_time()

    if arm == "A5_kernel":
        kernel = FidelityStatevectorKernel(feature_map=build_feature_map("ry", n))
        Ktr, Kval, Kte = kernel.evaluate(Xtr), kernel.evaluate(Xval, Xtr), kernel.evaluate(Xte, Xtr)
        best = None
        for Cc in d["kernel_C"]:
            m = SVC(kernel="precomputed", C=Cc).fit(Ktr, ytr)
            f1 = f1_score(yval, m.predict(Kval), average="macro")
            if best is None or f1 > best[0]:
                best = (f1, Cc, m)
        m = best[2]
        pred_tr, pred_val, pred_te = m.predict(Ktr), m.predict(Kval), m.predict(Kte)
        train_loss, n_params, nfev, weights = np.nan, 0, 0, []
        extra["hyperparameter"] = f"C={best[1]}"
    elif arm == "R2_head_only":
        # <Z_i> of the RY encoding alone is cos(x_i); same head, loss, optimiser, budget rule.
        Z = {k: np.cos(v) for k, v in [("tr", Xtr), ("val", Xval), ("te", Xte)]}
        n_params = head_parameters(n, C)
        loss_fn = lambda w: mean_ce_bits(head_proba(Z["tr"], w, C), ytr)
        def objective(w):
            value = loss_fn(w)
            callback(w, value)
            return value
        res = COBYLA(maxiter=MODEL["evals_per_parameter"] * n_params).minimize(objective, np.zeros(n_params))
        w = np.asarray(res.x)
        pred_tr, pred_val, pred_te = (head_proba(Z[k], w, C).argmax(axis=1) for k in ("tr", "val", "te"))
        train_loss, nfev, weights = loss_fn(w), int(res.nfev), w
    elif arm == "A2_readout":
        model = ReadoutHeadModel(n, C, MODEL["reps"])
        x0 = np.concatenate([np.random.default_rng(seed).uniform(0, 2 * np.pi, model.n_circuit),
                             np.zeros(model.n_head)])
        model.fit(Xtr, ytr, MODEL["evals_per_parameter"] * model.num_parameters, x0, callback)
        pred_tr, pred_val, pred_te = model.predict(Xtr), model.predict(Xval), model.predict(Xte)
        train_loss = mean_ce_bits(model.proba(Xtr, model.weights), ytr)
        n_params, nfev, weights = model.num_parameters, int(model.result.nfev), model.weights
        extra["hyperparameter"] = f"circuit={model.n_circuit}, head={model.n_head}"
    else:
        reps = d["deeper_reps"] if arm == "A3_deeper" else MODEL["reps"]
        n_params = build_ansatz(MODEL["ansatz"], n, reps).num_parameters
        rng = np.random.default_rng(seed) if start == 0 else np.random.default_rng([seed, start])
        x0 = rng.uniform(0, 2 * np.pi, n_params)
        opt = COBYLA(maxiter=MODEL["evals_per_parameter"] * n_params)
        if arm == "A4_reupload":
            qc, x, th = reupload_circuit(n, reps, reupload=True)
            model = classifier_from_circuit(qc, x, th, n, C, opt, x0, callback)
        else:   # A0, A1, A3: the main-study builder, unchanged
            model = build_vqc("ry", MODEL["ansatz"], n, reps, C, optimizer=opt,
                              initial_point=x0, callback=callback)
        model.fit(Xtr, ytr)
        pred_tr, pred_val, pred_te = model.predict(Xtr), model.predict(Xval), model.predict(Xte)
        probs = model.predict_proba(Xtr)
        train_loss = mean_ce_bits(probs, ytr)
        nfev, weights = int(model.fit_result.nfev), np.asarray(model.fit_result.x)
        # Our loss function must agree with qiskit's own training objective at the final weights.
        if not np.isclose(train_loss, float(model.fit_result.fun), atol=1e-6):
            raise RuntimeError(f"loss mismatch {train_loss} vs {model.fit_result.fun} in {job_key(j)}")

    secs, cpu = time.time() - t0, time.process_time() - c0
    pred_tr, pred_val, pred_te = (np.asarray(p).ravel().astype(int) for p in (pred_tr, pred_val, pred_te))
    uniform = float(np.log2(C))
    row = {"task": task_id(sc, seed), "scenario": sc, "n_train": d["n_train"], "n_qubits": n,
           "reducer": d["reducer"], "sampling": d["sampling"], "seed": seed, "arm": arm, "start": start,
           "seconds": round(secs, 1), "cpu_seconds": round(cpu, 1)}
    row.update({f"test_{k}": v for k, v in metrics(yte, pred_te, C).items()})
    row.update({f"val_{k}": v for k, v in metrics(yval, pred_val, C).items()})
    row.update({"train_macro_f1": round(f1_score(ytr, pred_tr, average="macro"), 4),
                "final_train_loss": round(train_loss, 4) if not np.isnan(train_loss) else np.nan,
                "train_loss_bits": train_loss, "uniform_loss_bits": uniform,
                "loss_ratio": train_loss / uniform, "loss_evaluations": nfev, "parameters": n_params,
                "weights": json.dumps(np.round(np.asarray(weights, dtype=float), 6).tolist()), **extra})
    if losses:
        with open(f"{curves_dir}/{task_id(sc, seed)}_{arm}_start{start}.json", "w") as fh:
            json.dump(losses, fh)
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            if fh.readline().rstrip("\n").split(",") != COLUMNS:
                raise RuntimeError(f"{out_csv} has a different column layout")
    pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", index=False,
                                                       header=not os.path.exists(out_csv))
    print({k: row[k] for k in ["task", "arm", "start", "train_macro_f1", "loss_ratio",
                               "val_macro_f1", "test_macro_f1", "seconds"]}, flush=True)


# ------------------------------------------------------------------ self-checks
def check():
    rng = np.random.default_rng(7)
    n = 4
    # 1. Without re-uploading, our circuit is exactly the main-study circuit (same state for the
    #    same inputs and weights, in qiskit's parameter order).
    qc0, _, _ = reupload_circuit(n, MODEL["reps"], reupload=False)
    ref = QuantumCircuit(n)
    ref.compose(build_feature_map("ry", n), inplace=True)
    ref.compose(build_ansatz(MODEL["ansatz"], n, MODEL["reps"]), inplace=True)
    assert ANSATZ_ENTANGLEMENT == "linear"
    assert [p.name for p in qc0.parameters] == [p.name for p in ref.parameters]
    for _ in range(5):
        v = rng.uniform(0, 2 * np.pi, ref.num_parameters)
        assert np.allclose(Statevector(qc0.assign_parameters(v)).data,
                           Statevector(ref.assign_parameters(v)).data), "circuit mismatch"
    qc1, _, th = reupload_circuit(n, MODEL["reps"], reupload=True)
    print(f"[ok] A4 circuit without repeats == main-study circuit; with repeats: "
          f"{len(th)} weights (same as A0), {(qc1.count_ops().get('ry', 0) - len(th) // 2) // n} encoding layers")
    # 2. qiskit's fidelity kernel equals the closed form for the RY product encoding.
    A, B = rng.uniform(0, np.pi, (6, n)), rng.uniform(0, np.pi, (5, n))
    K = FidelityStatevectorKernel(feature_map=build_feature_map("ry", n)).evaluate(A, B)
    assert np.allclose(K, ry_fidelity_kernel(A, B), atol=1e-10), "kernel mismatch"
    print("[ok] A5 fidelity kernel == prod cos^2((x - x')/2)")
    # 3. A2: <Z_i> from bitstring probabilities equals the statevector expectation; zero head
    #    gives the uniform guess (loss = log2 C bits).
    for C in [2, 8]:
        m = ReadoutHeadModel(n, C, MODEL["reps"])
        X = rng.uniform(0, np.pi, (3, n))
        w = np.concatenate([rng.uniform(0, 2 * np.pi, m.n_circuit), np.zeros(m.n_head)])
        z = m.expectations(X, w[:m.n_circuit])
        from qiskit.quantum_info import SparsePauliOp
        for i in range(3):
            sv = Statevector(m.qnn.circuit.assign_parameters(list(X[i]) + list(w[:m.n_circuit])))
            for q in range(n):
                label = "".join("Z" if k == q else "I" for k in reversed(range(n)))
                assert np.isclose(sv.expectation_value(SparsePauliOp(label)).real, z[i, q])
        y = rng.integers(0, C, 3)
        assert np.isclose(mean_ce_bits(m.proba(X, w), y), np.log2(C))
        print(f"[ok] A2 {C}-class: <Z_i> correct, {m.n_circuit}+{m.n_head} parameters, "
              f"zero head = uniform guess ({np.log2(C):.0f} bit{'s' if C > 2 else ''})")
    # 3b. R2: <Z_i> of the RY encoding alone is cos(x_i).
    X = rng.uniform(0, np.pi, (3, n))
    enc = build_feature_map("ry", n)
    for i in range(3):
        pz = Statevector(enc.assign_parameters(X[i])).probabilities()
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        assert np.allclose(pz @ (1 - 2 * bits), np.cos(X[i]))
    print("[ok] R2 <Z_i> of the encoding alone == cos(x_i)")
    # 4. A0 is built by the main-study code path and our loss equals qiskit's objective.
    tr, te, _ = load_scenario("Recon", 10, 0)
    Xtr, _, ytr, _, _ = to_qubits(tr, te, reducer="pca", n_qubits=n, angle_max=ANGLE_MAX["ry"], seed=0)
    p = build_ansatz(MODEL["ansatz"], n, MODEL["reps"]).num_parameters
    v = build_vqc("ry", MODEL["ansatz"], n, MODEL["reps"], 2, optimizer=COBYLA(maxiter=20),
                  initial_point=np.random.default_rng(0).uniform(0, 2 * np.pi, p)).fit(Xtr, ytr)
    assert np.isclose(mean_ce_bits(v.predict_proba(Xtr), ytr), v.fit_result.fun, atol=1e-6)
    print(f"[ok] loss in bits equals qiskit's training objective ({v.fit_result.fun:.4f})")
    print(f"jobs: {len(jobs())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the self-checks only")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run into results/diagnostic_smoke")
    ap.add_argument("--plan", action="store_true", help="count jobs and estimate time only")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.check:
        check()
        raise SystemExit
    if args.smoke:
        use_smoke_settings()

    all_jobs = jobs()
    curves_dir = f"{OUT}/curves"
    os.makedirs(curves_dir, exist_ok=True)
    done = set()
    for f in glob.glob(f"{OUT}/diag_shard*.csv"):
        dd = pd.read_csv(f, usecols=["task", "arm", "start"])
        done |= {f"{t}|{a}|{s}" for t, a, s in zip(dd.task, dd.arm, dd.start)}
    owner, load = assign_shards(all_jobs, args.nshards)

    if args.plan:
        todo = [j for j in all_jobs if job_key(j) not in done]
        hours = sum(estimate_seconds(j) for j in todo) / 3600
        per = [sum(estimate_seconds(j) for j in todo if owner[job_key(j)] == k) / 3600
               for k in range(args.nshards)]
        print(f"{len(all_jobs)} jobs ({len(all_jobs) - len(todo)} done); about {hours:.1f} h on one "
              f"process -> {max(per):.1f} h with {args.nshards} parallel shards (rough estimate)")
        print(pd.DataFrame([{"arm": j[2], "h": estimate_seconds(j) / 3600} for j in todo])
              .groupby("arm").h.sum().round(1).to_string())
        raise SystemExit

    with open(f"{OUT}/diag_config.json", "w") as fh:
        json.dump({"diag": DIAG, "model": MODEL, "n_val_per_class": N_VAL_PER_CLASS}, fh, indent=2)
    mine = [j for j in all_jobs if owner[job_key(j)] == args.shard and job_key(j) not in done]
    out_csv = f"{OUT}/diag_shard{args.shard}of{args.nshards}.csv"
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} jobs")
    for i, j in enumerate(mine, 1):
        print(f"--- [{i}/{len(mine)}] {job_key(j)}", flush=True)
        run_job(j, out_csv, curves_dir)
