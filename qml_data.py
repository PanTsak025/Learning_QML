"""Builds the dataset for one QML experiment from the pools made by 03_build_pools.py.

    train, test, classes = load_scenario("Recon", n_train_per_class=100, seed=0)
    X_tr, X_te, y_tr, y_te, info = to_qubits(train, test, reducer="pca", n_qubits=4)
"""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


OUT = "prepared"
FEAT = json.load(open(f"{OUT}/features.json"))
KEPT, LOG_COLS = FEAT["kept"], FEAT["log1p"]
MI_MAX_CORR = 0.95   # MI selection skips features with |Spearman rho| >= this to an already selected one
NEG_TO_ZERO = FEAT.get("negative_to_zero", [])
N_TEST_PER_CLASS = json.load(open(f"{OUT}/pools/pool_config.json"))["test_per_family"]

ATTACKS = ["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "BruteForce", "Web"]
# 7 x "Benign vs one family", plus Benign vs all attacks, plus the 8-class problem.
SCENARIOS = ATTACKS + ["benign_vs_attack", "multiclass"]


def scenario_classes(scenario):
    """Class names in label order. Benign is always class 0."""
    if scenario == "multiclass":
        return ["Benign"] + ATTACKS
    if scenario == "benign_vs_attack":
        return ["Benign", "Attack"]
    if scenario in ATTACKS:
        return ["Benign", scenario]
    raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")


@lru_cache(maxsize=None)
def _pool(split, fam):
    return pd.read_parquet(f"{OUT}/pools/{split}_{fam}.parquet")


def _split_evenly(n, k):
    """n samples over k families as evenly as possible, e.g. 25 over 7 -> 4,4,4,4,3,3,3."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def _log_standardise(X):
    X = X.astype("float64").copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
    return StandardScaler().fit_transform(X)


def _kmeans_prototypes(pool, n, seed):
    """Representative sampling: cluster the class's candidate pool into n clusters and keep
    the real sample closest to each cluster centre. The n samples then cover the whole
    class distribution rather than mostly its densest region."""
    Z = _log_standardise(pool[KEPT])        # same feature scaling as the experiments
    km = KMeans(n_clusters=n, n_init=1, random_state=seed).fit(Z)
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    pick = pd.Series(dist).groupby(km.labels_).idxmin().to_numpy()
    if len(pick) < n:   # only if k-means leaves a cluster empty (rare): top up at random
        rest = pool.index.difference(pick).to_series().sample(n - len(pick), random_state=seed)
        pick = np.concatenate([pick, rest.to_numpy()])
    return pool.loc[pick]


def _draw(split, families, n, seed, strategy):
    parts = []
    for fam, k in zip(families, _split_evenly(n, len(families))):
        pool = _pool(split, fam)
        if k > len(pool):
            raise ValueError(f"{split} pool of {fam} has {len(pool)} rows, {k} requested")
        if split == "test":
            parts.append(pool.head(k))       # pool is already randomly ordered -> fixed test set
        elif strategy == "random":
            # Shuffle with the seed, then take the first k. For one seed, smaller training
            # sets are subsets of larger ones (nested), so size effects aren't sampling noise.
            parts.append(pool.sample(frac=1, random_state=seed).head(k))
        elif strategy == "kmeans":
            parts.append(_kmeans_prototypes(pool, k, seed))
        else:
            raise ValueError("strategy must be 'random' or 'kmeans'")
    return pd.concat(parts, ignore_index=True)


def load_scenario(scenario, n_train_per_class, seed, strategy="random"):
    """Balanced training set (n per class) and the fixed test set (N_TEST_PER_CLASS per class).
    Returns DataFrames with the features, 'Label' (subtype), 'row_id' and 'y' (class index)."""
    classes = scenario_classes(scenario)
    train, test = [], []
    for y, cls in enumerate(classes):
        fams = ATTACKS if cls == "Attack" else [cls]   # 'Attack' = equal share of all 7 families
        train.append(_draw("train", fams, n_train_per_class, seed, strategy).assign(y=y))
        test.append(_draw("test", fams, N_TEST_PER_CLASS, None, None).assign(y=y))
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True), classes


def to_qubits(train, test, reducer="pca", n_qubits=4, clip_pct=(1, 99), angle_max=np.pi, seed=42):
    """Feature pipeline. Every step is fitted on the TRAINING set only, then applied to test:
    log1p (rule 7) -> standardise -> reduce to n_qubits (PCA or MI) -> angle scale to [0, angle_max]."""
    Xtr = train[KEPT].astype("float64").copy()
    Xte = test[KEPT].astype("float64").copy()
    Xtr[NEG_TO_ZERO] = Xtr[NEG_TO_ZERO].clip(lower=0)   # Rule 8
    Xte[NEG_TO_ZERO] = Xte[NEG_TO_ZERO].clip(lower=0)
    y_tr = train["y"].to_numpy()

    Xtr[LOG_COLS] = np.log1p(Xtr[LOG_COLS])
    Xte[LOG_COLS] = np.log1p(Xte[LOG_COLS].clip(lower=0))   # guard against unseen negatives

    # A feature can be constant within a small training sample (e.g. SMTP = 0 everywhere);
    # it can't be standardised and carries no information there, so it is skipped.
    cols = Xtr.columns[Xtr.std() > 0]
    scaler = StandardScaler().fit(Xtr[cols])
    Ztr, Zte = scaler.transform(Xtr[cols]), scaler.transform(Xte[cols])

    info = {"reducer": reducer, "n_qubits": n_qubits}
    if reducer == "pca":
        # Unsupervised: keeps the n_qubits directions of largest variance.
        pca = PCA(n_components=n_qubits, random_state=seed).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
        info["explained_variance"] = float(pca.explained_variance_ratio_.sum())
        info["loadings"] = pd.DataFrame(pca.components_.T, index=cols,
                                        columns=[f"PC{i+1}" for i in range(n_qubits)])
    elif reducer == "mi":
        # Supervised: rank features by mutual information with the class label, then pick
        # greedily from the top, skipping any feature strongly correlated with one already
        # chosen, so each qubit carries different information (mRMR idea, Peng et al. 2005).
        mi = mutual_info_classif(Ztr, y_tr, random_state=seed)
        rho = np.abs(pd.DataFrame(Ztr).corr(method="spearman").to_numpy())
        order = np.argsort(mi)[::-1]
        top = []
        for j in order:
            if all(rho[j, t] < MI_MAX_CORR for t in top):
                top.append(j)
            if len(top) == n_qubits:
                break
        for j in order:                      # fallback: too few non-redundant features
            if len(top) == n_qubits:
                break
            if j not in top:
                top.append(j)
        top = np.array(top)
        Ztr, Zte = Ztr[:, top], Zte[:, top]
        info["features"] = list(cols[top])
        info["mi_scores"] = [float(mi[i]) for i in top]
        info["mi_max_corr"] = MI_MAX_CORR
    else:
        raise ValueError("reducer must be 'pca' or 'mi'")

    # Angle scaling: map the training 1st..99th percentile range to [0, pi] and clip outside
    # values to the edges. A plain min-max would let a few extreme values squeeze all other
    # samples into a tiny angle range. For a sparse feature whose percentiles coincide
    # (e.g. 99 % zeros), the full min..max range is used instead.
    lo, hi = np.percentile(Ztr, clip_pct, axis=0)
    flat = hi <= lo
    lo[flat], hi[flat] = Ztr[:, flat].min(axis=0), Ztr[:, flat].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Ztr = angle_max * np.clip((Ztr - lo) / span, 0, 1)
    Zte = angle_max * np.clip((Zte - lo) / span, 0, 1)
    info["clip_percentiles"] = clip_pct
    info["angle_max"] = angle_max
    return Ztr, Zte, y_tr, test["y"].to_numpy(), info


if __name__ == "__main__":
    print("Rule 8 applied to:", NEG_TO_ZERO, "| IAT log1p:", "IAT" in LOG_COLS)
    # Self-check: every scenario builds, values are in [0, pi], sets are balanced and nested.
    for sc in SCENARIOS:
        tr, te, classes = load_scenario(sc, n_train_per_class=100, seed=0)
        Xtr, Xte, ytr, yte, info = to_qubits(tr, te, reducer="pca", n_qubits=4)
        print(f"{sc:<17} train={Xtr.shape} test={Xte.shape} "
              f"PCA var={info['explained_variance']:.1%} "
              f"range=[{Xtr.min():.2f}, {Xtr.max():.2f}]")
    tr, te, _ = load_scenario("Recon", 100, 0, strategy="kmeans")
    _, _, _, _, info = to_qubits(tr, te, reducer="mi", n_qubits=4)
    print("Recon, kmeans + MI features:", info["features"])
    small, _, _ = load_scenario("DDoS", 50, 0)
    large, _, _ = load_scenario("DDoS", 100, 0)
    print("nested training sets:", set(small.row_id) <= set(large.row_id))

def load_validation(scenario, n_per_class, exclude_row_ids, seed=12345):
    """Validation set for tuning decisions (optimizer, ansatz, depth).
    Drawn from the TRAINING pools, so the test set is never used for any choice; rows in
    `exclude_row_ids` (the training sample) are excluded, so validation is disjoint from it."""
    exclude = set(exclude_row_ids)
    classes = scenario_classes(scenario)
    parts = []
    for y, cls in enumerate(classes):
        fams = ATTACKS if cls == "Attack" else [cls]
        for fam, k in zip(fams, _split_evenly(n_per_class, len(fams))):
            pool = _pool("train", fam)
            pool = pool[~pool["row_id"].isin(exclude)]
            parts.append(pool.sample(n=k, random_state=seed).assign(y=y))
    return pd.concat(parts, ignore_index=True), classes