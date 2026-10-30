"""
CIC-IoT-2023 preprocessing for QML (merged CSVs).

PART 1 (cleaning, Rules 1-4) reads all 63 CSVs twice and writes prepared/. It is slow and
only needs to run once, so it is commented out. To reproduce prepared/ from scratch,
uncomment PART 1 and comment out the "LOAD SAVED RESULTS" block below it.

PART 2 (feature decisions, Rules 5-8) and PART 3 (PCA exploration) run from the saved files.
"""
import glob, json, os, platform, shutil, sys
import numpy as np
import pandas as pd
import sklearn

# Every tunable choice lives here and is saved to disk, so a reader can rerun the same pipeline.
CFG = {
    "seed": 42,                                # one seed for all random operations
    "label_col": "Label",
    "test_hash_modulus": 5,                    # a row goes to the test pool if hash % 5 == 0  -> ~20 %
    "analysis_rows_per_family_per_file": 500,  # size of the sample used for feature analysis only
    "redundancy_spearman_threshold": 0.999,    # |rho| at or above this = same information
}
OUT = "prepared"
os.makedirs(OUT, exist_ok=True)

VERSIONS = {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "platform": platform.platform()}
with open(f"{OUT}/config.json", "w") as fh:
    json.dump({"config": CFG, "versions": VERSIONS}, fh, indent=2)

SEED, LABEL = CFG["seed"], CFG["label_col"]


# 34 labels -> 8 families, following the CICIoT2023 paper's attack categories.
def family(label):
    s = label.upper()
    if s == "BENIGN":                                     return "Benign"
    if s.startswith("DDOS"):                              return "DDoS"
    if s.startswith("DOS"):                               return "DoS"
    if s.startswith("MIRAI"):                             return "Mirai"
    if s.startswith("RECON") or s == "VULNERABILITYSCAN": return "Recon"
    if "SPOOFING" in s:                                   return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":                       return "BruteForce"
    return "Web"   # BrowserHijacking, SQLInjection, CommandInjection, XSS, Backdoor, Uploading


files = sorted(glob.glob(os.path.join("CIC-IOT-2023", "*.csv")))   # sorted -> same order on every machine
print(len(files), "files")


# =====================================================================================
# PART 1: CLEANING (Rules 1-4). Already run; output is in prepared/. Uncomment to rerun.
# =====================================================================================
# BEGIN PART 1
# # ---- Pass 1: compact index of every row in the dataset ----
# # Whole dataset is ~45M rows; loading all 63 files at once would need ~15 GB RAM.
# # Instead we keep, per row: file, row number, a 64-bit hash of its features, and its label.
# # That is ~15 bytes per row (~1-3 GB in total), enough to find duplicates across ALL files.
# # Chance of two different rows sharing a 64-bit hash with ~45M rows is ~5e-5: negligible.
# COLUMNS = None
# label_code = {}
# index_parts, raw_counts, bad_counts = [], [], []
#
# for i, f in enumerate(files):
#     d = pd.read_csv(f)
#     if COLUMNS is None:
#         COLUMNS = list(d.columns)
#         FEATURES = [c for c in COLUMNS if c != LABEL]
#     assert list(d.columns) == COLUMNS, f"{f} has different columns"
#     for lab in d[LABEL].unique():
#         label_code.setdefault(lab, len(label_code))
#
#     # Cast to float64 before hashing: a column read as int in one file and float in
#     # another must still hash identically, or cross-file duplicates would be missed.
#     X = d[FEATURES].astype("float64")
#
#     # Rule 1: rows with NaN or inf in any feature are removed (no valid value to learn from).
#     bad = (X.isna() | np.isinf(X)).any(axis=1).to_numpy()
#     raw_counts.append(d[LABEL].value_counts())
#     bad_counts.append(d.loc[bad, LABEL].value_counts())
#
#     good = ~bad
#     index_parts.append(pd.DataFrame({
#         "file":  np.int16(i),
#         "row":   np.flatnonzero(good).astype(np.int32),
#         "fhash": pd.util.hash_pandas_object(X[good], index=False).to_numpy(),
#         "label": d.loc[good, LABEL].map(label_code).to_numpy(np.int8),
#     }))
#     print(f"[{i+1}/{len(files)}] {f}: {len(d):,} rows, {bad.sum()} with NaN/inf")
#
# idx = pd.concat(index_parts, ignore_index=True)
# del index_parts   # free memory: the per-file tables are no longer needed
#
# # Rule 2: exact duplicates (same features AND same label) are kept once, at the first
# # occurrence in sorted file order. Otherwise a copy could land in both train and test.
# dup = idx.duplicated(subset=["fhash", "label"], keep="first")
# dup_counts = idx.loc[dup, "label"].value_counts()
# idx = idx[~dup]
#
# # Rule 3: if one feature vector appears with more than one label, the ground truth is
# # ambiguous and no classifier can be right on all copies, so every copy is removed.
# conflict = idx.duplicated(subset="fhash", keep=False)
# conflict_counts = idx.loc[conflict, "label"].value_counts()
# idx = idx[~conflict].copy()
#
# # Rule 4: train/test split by hash: test if hash % 5 == 0 (~20 %).
# # Deterministic: it depends only on the row's content, not on a seed or file order.
# # After rules 2-3 every feature vector is unique, so no row can be in both pools.
# idx["test"] = (idx["fhash"] % CFG["test_hash_modulus"]) == 0
#
# # Report tables for the paper: what was removed at each stage, per label and per family.
# names = {v: k for k, v in label_code.items()}
# report = pd.DataFrame({
#     "raw":       pd.concat(raw_counts, axis=1).sum(axis=1),
#     "nan_inf":   pd.concat(bad_counts, axis=1).sum(axis=1),
#     "duplicate": dup_counts.rename(index=names),
#     "conflict":  conflict_counts.rename(index=names),
#     "clean":     idx["label"].value_counts().rename(index=names),
#     "train":     idx.loc[~idx["test"], "label"].value_counts().rename(index=names),
#     "test":      idx.loc[idx["test"], "label"].value_counts().rename(index=names),
# }).fillna(0).astype("int64")
# report.insert(0, "family", report.index.map(family))
# report = report.sort_values(["family", "raw"], ascending=[True, False])
# report.loc["TOTAL"] = ["", *report.drop(columns="family").sum()]
# report.to_csv(f"{OUT}/cleaning_report_by_label.csv")
#
# fam = report.drop(index="TOTAL").groupby("family").sum(numeric_only=True)
# fam.loc["TOTAL"] = fam.sum()
# fam.to_csv(f"{OUT}/cleaning_report_by_family.csv")
# print(fam)
#
# # ---- Pass 2: write the clean train and test pools to disk ----
# # Clean rows are saved as Parquet, split into train/test and partitioned by family,
# # so later steps can read e.g. only "train, Web" without loading 15M+ rows.
# # Features stored as float32 (~7 significant digits): halves disk/RAM, ample for ML.
# for split in ["train", "test"]:
#     shutil.rmtree(f"{OUT}/{split}", ignore_errors=True)   # rerunning must not append twice
#
# rows_by_file = {i: g for i, g in idx.groupby("file")}
# analysis_parts = []
# stats = {"min": None, "max": None, "nonzero": 0, "n": 0}
#
# for i, f in enumerate(files):
#     d = pd.read_csv(f)
#     k = rows_by_file[i]
#     d = d.iloc[k["row"].to_numpy()].reset_index(drop=True)
#     d[FEATURES] = d[FEATURES].astype("float32")
#     d["family"] = d[LABEL].map(family)
#
#     is_test = k["test"].to_numpy()
#     d[is_test].to_parquet(f"{OUT}/test", partition_cols=["family"], index=False)
#     tr = d[~is_test]
#     tr.to_parquet(f"{OUT}/train", partition_cols=["family"], index=False)
#
#     # Exact per-feature statistics over the WHOLE training pool (used for rules 5 and 8).
#     mn, mx = tr[FEATURES].min(), tr[FEATURES].max()
#     stats["min"] = mn if stats["min"] is None else np.minimum(stats["min"], mn)
#     stats["max"] = mx if stats["max"] is None else np.maximum(stats["max"], mx)
#     stats["nonzero"] = stats["nonzero"] + (tr[FEATURES] != 0).sum()
#     stats["n"] += len(tr)
#
#     # Analysis sample: up to 500 random TRAIN rows per family per file. Family-balanced,
#     # so the feature analysis isn't dominated by DDoS (~60 % of all traffic). Train only:
#     # the test pool is never used for any decision.
#     analysis_parts.append(tr.sample(frac=1, random_state=SEED + i)
#                             .groupby("family").head(CFG["analysis_rows_per_family_per_file"]))
#     print(f"[{i+1}/{len(files)}] {f}: {(~is_test).sum():,} train / {is_test.sum():,} test")
#
# analysis = pd.concat(analysis_parts, ignore_index=True)
# analysis.to_parquet(f"{OUT}/analysis_sample.parquet", index=False)
#
# feature_stats = pd.DataFrame({"min": stats["min"], "max": stats["max"],
#                               "nonzero_pct": 100 * stats["nonzero"] / stats["n"]})
# feature_stats.to_csv(f"{OUT}/feature_stats_train.csv")   # raw stats, saved BEFORE rule 8
# END PART 1

# ---- LOAD SAVED RESULTS (comment this block out if you uncomment PART 1) ----
# BEGIN LOAD
analysis = pd.read_parquet(f"{OUT}/analysis_sample.parquet")
feature_stats = pd.read_csv(f"{OUT}/feature_stats_train.csv", index_col=0)
FEATURES = list(feature_stats.index)
# END LOAD

print(analysis["family"].value_counts())


# =====================================================================================
# PART 2: FEATURE DECISIONS (Rules 5-8). Label-free, training data only.
# =====================================================================================

# Rule 8: all features here are non-negative by definition (counts, sizes, times, rates,
# proportions). Negative values can only be measurement errors, so they are set to 0.
# Applies only to features that actually have negatives in the training pool (IAT).
# Applied first, because Rule 7 below only log-transforms non-negative features.
NEG_TO_ZERO = feature_stats.index[feature_stats["min"] < 0].tolist()
print("negative values set to 0 in:", NEG_TO_ZERO)
analysis[NEG_TO_ZERO] = analysis[NEG_TO_ZERO].clip(lower=0)
feature_stats.loc[NEG_TO_ZERO, "min"] = 0.0

# Rule 5: a feature that is constant over the whole training pool carries no information.
constant = feature_stats.index[feature_stats["min"] == feature_stats["max"]].tolist()
print("constant in training pool:", constant)

# Rule 6: two features with |Spearman rho| >= 0.999 carry the same information.
# Spearman (rank) correlation detects any monotonic relation, e.g. Variance = Std^2.
# Of each pair we keep the column that comes first in the original CSV order.
candidates = [c for c in FEATURES if c not in constant]
rho = analysis[candidates].corr(method="spearman").abs()
redundant, pairs = [], []
for a_i, a in enumerate(candidates):
    for b in candidates[a_i + 1:]:
        if rho.loc[a, b] >= CFG["redundancy_spearman_threshold"]:
            pairs.append((a, b, round(float(rho.loc[a, b]), 5)))
            if a not in redundant and b not in redundant:
                redundant.append(b)
print("redundant pairs (kept, dropped, |rho|):", pairs)

# For information only: strongly but not perfectly correlated pairs. These are NOT
# dropped; the qubit-reduction step (PCA / MI selection) deals with them.
upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
print(upper[(upper >= 0.9) & (upper < CFG["redundancy_spearman_threshold"])]
      .sort_values(ascending=False))

KEPT = [c for c in candidates if c not in redundant]
print(len(KEPT), "features kept")

# Rule 7: log1p is applied to a non-negative feature only if it reduces the absolute
# skewness on the training analysis sample. Data-driven, with no hand-picked threshold.
skew_before = analysis[KEPT].skew()
skew_after = np.log1p(analysis[KEPT].clip(lower=0)).skew()
nonneg = feature_stats.loc[KEPT, "min"] >= 0
LOG_COLS = [c for c in KEPT if nonneg[c] and abs(skew_after[c]) < abs(skew_before[c])]

skew_table = pd.DataFrame({"skew_raw": skew_before, "skew_log1p": skew_after,
                           "log1p_applied": [c in LOG_COLS for c in KEPT]}).round(2)
skew_table.to_csv(f"{OUT}/skewness_train.csv")
print(skew_table)

# Save every feature decision; the next steps read this file instead of hard-coding lists.
with open(f"{OUT}/features.json", "w") as fh:
    json.dump({"all_features": FEATURES, "constant": constant, "redundant_pairs": pairs,
               "dropped": constant + redundant, "kept": KEPT, "log1p": LOG_COLS,
               "negative_to_zero": NEG_TO_ZERO},
              fh, indent=2)


# =====================================================================================
# PART 3: PCA EXPLORATION (exploratory only; experiments fit PCA per scenario on train data)
# =====================================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = analysis[KEPT].astype("float64").copy()
Z[LOG_COLS] = np.log1p(Z[LOG_COLS])          # 1. compress heavy tails
Z = StandardScaler().fit_transform(Z)        # 2. mean 0, std 1 -> no feature wins by scale
pca = PCA(random_state=SEED).fit(Z)          # 3. all components, to see the full picture

# How much information k qubits would keep:
cum = np.cumsum(pca.explained_variance_ratio_)
print(pd.DataFrame({"k_qubits": np.arange(1, 13),
                    "variance_%": (100 * pca.explained_variance_ratio_[:12]).round(1),
                    "cumulative_%": (100 * cum[:12]).round(1)}).to_string(index=False))

# Which original features build each of the first 4 components (top 5 by |weight|):
loadings = pd.DataFrame(pca.components_[:4].T, index=KEPT,
                        columns=["PC1", "PC2", "PC3", "PC4"])
for pc in loadings:
    top = loadings[pc].abs().sort_values(ascending=False).head(5).index
    print(pc, ", ".join(f"{c} ({loadings.loc[c, pc]:+.2f})" for c in top))