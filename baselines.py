"""Classical baselines, trained on exactly the same samples as the VQC.

Two kinds:
  * same features : the SAME n_qubits features the VQC sees (after PCA / MI and angle
                    scaling). Answers: "with identical information, does the VQC do
                    better or worse than a classical model?"
  * all features  : all 33 kept features (Rules 5-8, log1p, standardised). A reference
                    for how much information the n_qubits-feature bottleneck throws away.
Each model's main regularisation hyperparameter is chosen on the same VALIDATION set used
for the QML pilots (never the test set), so the classical models are tuned as fairly as
the VQC was.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_data import KEPT, LOG_COLS, NEG_TO_ZERO

# Model -> (constructor, small grid for its main regularisation hyperparameter).
CANDIDATES = {
    "LogReg":  (lambda p, s: LogisticRegression(C=p, max_iter=2000),          [0.1, 1, 10]),
    "SVM-RBF": (lambda p, s: SVC(C=p, kernel="rbf", gamma="scale"),         [0.1, 1, 10]),
    "RF":      (lambda p, s: RandomForestClassifier(n_estimators=200, max_depth=p,
                                                    random_state=s, n_jobs=1), [5, 10, None]),
}


def all_features(train, *others):
    """All 33 kept features with Rule 8 + log1p + standardisation, fitted on train only."""
    def prep(df):
        X = df[KEPT].astype("float64").copy()
        X[NEG_TO_ZERO] = X[NEG_TO_ZERO].clip(lower=0)
        X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
        return X
    scaler = StandardScaler().fit(prep(train))
    return [scaler.transform(prep(d)) for d in (train, *others)]


def fit_tuned(name, Xtr, ytr, Xval, yval, seed):
    """Pick the hyperparameter with the best validation macro-F1, return the fitted model."""
    make, grid = CANDIDATES[name]
    best = None
    for p in grid:
        model = make(p, seed).fit(Xtr, ytr)
        f1 = f1_score(yval, model.predict(Xval), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, p, model)
    return best[2], best[1]