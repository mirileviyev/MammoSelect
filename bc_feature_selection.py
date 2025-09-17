# bc_feature_selection.py

from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.base import ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Optional UCI fetch (only used when --use_builtin is passed)
try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except Exception:
    UCI_AVAILABLE = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --------------------------
# Utilities
# --------------------------


def as_float_2d(a) -> np.ndarray:
    arr = cast(np.ndarray, np.asarray(a, dtype=float))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got {arr.ndim}D with shape {arr.shape!r}")
    return arr


def as_bool_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=bool).ravel()


def sanitize_mask(mask, n_features, rng: np.random.Generator | None = None) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size < n_features:
        m = np.pad(m, (0, n_features - m.size), constant_values=False)
    elif m.size > n_features:
        m = m[:n_features]
    if not m.any():
        rng = rng or np.random.default_rng(RANDOM_SEED)
        m[rng.integers(0, n_features)] = True
    return m


def safe_predict_proba(clf: ClassifierMixin, x: np.ndarray) -> np.ndarray | None:
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        if hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(x), dtype=float).ravel()
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
    except Exception as e:
        print(f"Warning: Could not get probabilities: {e}")
        return None
    return None


def compute_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall/Sensitivity": rec,
        "Specificity": spec,
        "F1": f1,
        "Kappa": kappa,
        "MAE": mae,
        "RMSE": rmse,
    }
    if y_proba is not None:
        try:
            p = np.clip(y_proba, 1e-15, 1 - 1e-15)
            ce = log_loss(y_true, np.transpose(np.vstack([1 - p, p])), labels=[0, 1])
            metrics["CrossEntropy"] = ce
        except Exception as e:
            print(f"Warning: Could not compute cross entropy: {e}")
    return metrics


def safe_fit(base_estimator: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> ClassifierMixin | None:
    """Clone + fit an estimator safely. Returns fitted estimator or None on failure."""
    try:
        est = clone(base_estimator)
        est.fit(X, y)
        return est
    except Exception as e:
        name = getattr(base_estimator, "__class__", type("X", (object,), {})).__name__
        print(f"    Warning: failed to fit {name}: {e}")
        return None


def mask_fitness(estimator, x, y, mask, cv, alpha: float = 0.98) -> float:
    """
    Fitness = alpha * (CV accuracy) + (1 - alpha) * (1 - (#selected / total)).
    Uses error_score=0.0 (so failures don't produce NaN or warning spam).
    """
    mask_arr = as_bool_1d(mask)
    if int(mask_arr.sum()) == 0:
        return 0.0

    x_arr = as_float_2d(x)
    x_sub = x_arr[:, mask_arr]
    y_arr = np.asarray(y, dtype=int).ravel()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FitFailedWarning)
            cv_scores = cross_val_score(
                clone(estimator),
                x_sub,
                y_arr,
                cv=cv,
                scoring="accuracy",
                error_score=0.0,   # <-- FIXED to avoid NaN means and warning spam
            )
        mean_acc = float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0
        sparsity_bonus = 1.0 - (int(mask_arr.sum()) / int(x_arr.shape[1]))
        return alpha * mean_acc + (1.0 - alpha) * sparsity_bonus
    except Exception as e:
        print(f"    Warning: Fitness calculation failed: {e}")
        return 0.0


# --------------------------
# Bat Algorithm (BA)
# --------------------------
class BAFeatureSelector:
    def __init__(
        self,
        n_bats=12,
        n_iter=40,
        fmin=0.0,
        fmax=2.0,
        loudness=0.9,
        pulse_rate=0.5,
        alpha=0.98,
        rng: np.random.Generator | None = None,
    ):
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.fmin = fmin
        self.fmax = fmax
        self.init_loudness = loudness
        self.init_pulse = pulse_rate
        self.alpha = alpha
        self.rng = rng or np.random.default_rng(RANDOM_SEED)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def select(self, estimator, x, y, cv) -> Tuple[np.ndarray, float]:
        x_arr = as_float_2d(x)
        n_features = int(x_arr.shape[1])

        bats = np.asarray(self.rng.random((self.n_bats, n_features)) < 0.5, dtype=bool)
        velocities = np.zeros_like(bats, dtype=float)
        loud = np.full(self.n_bats, self.init_loudness, dtype=float)
        pulse = np.full(self.n_bats, self.init_pulse, dtype=float)

        # Initial fitness
        fitnesses = np.array(
            [mask_fitness(estimator, x_arr, y, m, cv, self.alpha) for m in bats],
            dtype=float,
        )
        best_idx = int(np.argmax(fitnesses))
        best_mask = bats[best_idx].copy()
        best_fit = float(fitnesses[best_idx])

        # Main loop
        for _ in range(self.n_iter):
            for i in range(self.n_bats):
                freq = self.fmin + (self.fmax - self.fmin) * self.rng.random()
                xi = bats[i].astype(float)
                xbest = best_mask.astype(float)
                velocities[i] = velocities[i] + (xi - xbest) * freq

                prob_flip = self._sigmoid(velocities[i])
                candidate = bats[i].copy()
                flip_indices = self.rng.random(n_features) < prob_flip
                candidate[flip_indices] = ~candidate[flip_indices]

                if self.rng.random() < pulse[i]:
                    n_local = max(1, int(0.05 * n_features))
                    flip_idx = self.rng.choice(n_features, size=n_local, replace=False)
                    candidate[flip_idx] = ~candidate[flip_idx]

                if not candidate.any():
                    candidate[self.rng.integers(0, n_features)] = True

                cand_fit = mask_fitness(estimator, x_arr, y, candidate, cv, self.alpha)

                if (cand_fit > fitnesses[i]) and (self.rng.random() < loud[i]):
                    bats[i] = candidate
                    fitnesses[i] = cand_fit
                    loud[i] *= 0.9
                    pulse[i] = pulse[i] * 0.9 + 0.1

                if fitnesses[i] > best_fit:
                    best_fit = fitnesses[i]
                    best_mask = bats[i].copy()

        best_mask = sanitize_mask(best_mask, n_features, self.rng)
        return best_mask.astype(bool, copy=False), float(best_fit)


# --------------------------
# Imperialist Competitive Algorithm (ICA)
# --------------------------
class ICAFeatureSelector:
    def __init__(
        self,
        n_countries=20,
        n_imperialists=5,
        n_iter=40,
        assimilation_rate=0.5,
        revolution_prob=0.02,
        alpha=0.98,
        rng: np.random.Generator | None = None,
    ):
        assert n_imperialists < n_countries
        self.n_countries = n_countries
        self.n_imperialists = n_imperialists
        self.n_iter = n_iter
        self.assimilation_rate = assimilation_rate
        self.revolution_prob = revolution_prob
        self.alpha = alpha
        self.rng = rng or np.random.default_rng(RANDOM_SEED)

    def select(self, estimator, x, y, cv) -> Tuple[np.ndarray, float]:
        x_arr = as_float_2d(x)
        n_features = int(x_arr.shape[1])

        countries = np.asarray(self.rng.random((self.n_countries, n_features)) < 0.5, dtype=bool)
        fitnesses = np.array(
            [mask_fitness(estimator, x_arr, y, m, cv, self.alpha) for m in countries],
            dtype=float,
        )
        costs = 1.0 - fitnesses

        order = np.argsort(costs)
        imperialists = countries[order[:self.n_imperialists]].copy()
        imp_costs = costs[order[:self.n_imperialists]]
        colonies = countries[order[self.n_imperialists:]].copy()
        col_costs = costs[order[self.n_imperialists:]]

        inv_cost = 1.0 / (imp_costs + 1e-9)
        shares = inv_cost / inv_cost.sum()
        n_colonies = len(colonies)
        alloc = (shares * n_colonies).astype(int)
        while alloc.sum() < n_colonies:
            alloc[np.argmin(alloc)] += 1
        while alloc.sum() > n_colonies:
            alloc[np.argmax(alloc)] -= 1

        empires, start = [], 0
        for k in range(self.n_imperialists):
            num = int(alloc[k])
            empires.append(list(range(start, start + num)))
            start += num

        best_imp_idx = int(np.argmin(imp_costs))
        best_mask = imperialists[best_imp_idx].copy()
        best_fit = 1.0 - float(imp_costs[best_imp_idx])

        for _ in range(self.n_iter):
            for k in range(self.n_imperialists):
                imp = imperialists[k]
                for ci in empires[k]:
                    col = colonies[ci]

                    diff = np.logical_xor(imp, col)
                    idx_diff = np.flatnonzero(diff)
                    new_col = col.copy()

                    if len(idx_diff) > 0:
                        copy_mask = self.rng.random(len(idx_diff)) < self.assimilation_rate
                        idx_to_copy = idx_diff[copy_mask]
                        if len(idx_to_copy) > 0:
                            new_col[idx_to_copy] = imp[idx_to_copy]

                    rev_mask = self.rng.random(n_features) < self.revolution_prob
                    new_col[rev_mask] = ~new_col[rev_mask]

                    if not new_col.any():
                        new_col[self.rng.integers(0, n_features)] = True

                    new_fit = mask_fitness(estimator, x_arr, y, new_col, cv, self.alpha)
                    new_cost = 1.0 - new_fit

                    if new_cost < col_costs[ci]:
                        colonies[ci] = new_col
                        col_costs[ci] = new_cost

                    if col_costs[ci] < imp_costs[k]:
                        imperialists[k], colonies[ci] = new_col, imp.copy()
                        imp_costs[k], col_costs[ci] = new_cost, imp_costs[k]

            total_costs = np.array(
                [
                    imp_costs[k] + (col_costs[empires[k]].mean() if empires[k] else 0.0)
                    for k in range(self.n_imperialists)
                ]
            )
            best_emp = int(np.argmin(total_costs))
            worst_emp = int(np.argmax(total_costs))
            if empires[worst_emp] and worst_emp != best_emp:
                worst_colony_idx = empires[worst_emp][
                    np.argmax([col_costs[ci] for ci in empires[worst_emp]])
                ]
                empires[worst_emp].remove(worst_colony_idx)
                empires[best_emp].append(worst_colony_idx)

            current_best = int(np.argmin(imp_costs))
            if imp_costs[current_best] < (1.0 - best_fit):
                best_fit = 1.0 - imp_costs[current_best]
                best_mask = imperialists[current_best].copy()

        best_mask = sanitize_mask(best_mask, n_features, self.rng)
        return best_mask.astype(bool, copy=False), float(best_fit)


# --------------------------
# Models
# --------------------------
def build_core_classifiers() -> Dict[str, ClassifierMixin]:
    return {
        "RF": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
        "LR": LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_SEED),
        "DT": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "NB": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "LDA": LinearDiscriminantAnalysis(),
        "ANN": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_SEED),
    }


def build_article1_classifiers() -> Dict[str, ClassifierMixin]:
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "LR": LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(500, 500, 500), activation="relu", max_iter=3000, random_state=RANDOM_SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# --------------------------
# Data Loading
# --------------------------
def load_wdbc_builtin() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Built-in WDBC fallback from sklearn or UCI."""
    if UCI_AVAILABLE:
        try:
            ds = fetch_ucirepo(id=17)  # WDBC
            X = ds.data.features.values
            y = ds.data.targets.values.ravel()
            fn = list(ds.data.features.columns)
            if y.dtype == object or isinstance(y[0], str):
                y = np.array([1 if str(lbl).upper() == "M" else 0 for lbl in y], dtype=int)
            return X, y, fn
        except Exception:
            pass
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names)


def load_coimbra_builtin() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not UCI_AVAILABLE:
        raise ImportError("ucimlrepo required for Coimbra dataset. Install with: pip install ucimlrepo")
    ds = fetch_ucirepo(id=451)  # Coimbra
    X = ds.data.features.values
    y_raw = ds.data.targets.values.ravel()
    fn = list(ds.data.features.columns)
    # 1=healthy, 2=patient -> map to 0/1
    y = np.array([1 if int(v) == 2 else 0 for v in y_raw], dtype=int)
    return X, y, fn


def load_dataset_from_csv(path: str, dataset_type: str) -> Tuple[NDArray[np.float_], NDArray[np.int_], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)

    # Drop obvious ID columns if present
    id_cols = [c for c in df.columns if c.lower() in ["id", "unnamed: 0", "unnamed: 32"]]
    if id_cols:
        df = df.drop(columns=id_cols)

    dataset_type = dataset_type.lower()
    target_col = None

    if dataset_type == "wdbc":
        # Diagnosis: 'M' / 'B' or 1 / 0
        for c in ["Diagnosis", "diagnosis", "target", "class", "Class"]:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            raise ValueError("Could not find WDBC target column (expected 'Diagnosis').")
        y_raw = df[target_col]
        if y_raw.dtype == object:
            y = (
                y_raw.astype(str)
                .str.upper()
                .map({"M": 1, "B": 0})
                .fillna(y_raw)
                .astype(int)
                .values
            )
        else:
            y = y_raw.astype(int).values

    elif dataset_type == "coimbra":
        # Class or Classification: 1/2 -> 0/1
        for c in ["Class", "class", "Classification", "classification", "Diagnosis", "diagnosis", "target"]:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            raise ValueError("Could not find Coimbra target column (expected 'Class' or 'Classification').")
        y_raw = df[target_col].astype(str).str.extract(r"(\d+)").astype(int).iloc[:, 0].values
        y = np.array([1 if v == 2 else 0 for v in y_raw], dtype=int)

    else:
        raise ValueError("dataset_type must be 'wdbc' or 'coimbra'")

    X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)
    fn = list(X_df.columns)
    y = np.asarray(y, dtype=int).ravel()
    return X, y, fn


# --------------------------
# Experiment
# --------------------------
@dataclass
class ExperimentResult:
    dataset: str
    selector: str
    clf_name: str
    cv_fitness: float
    n_features: int
    feature_indices: List[int]
    test_metrics: Dict[str, float]


def run_experiment_on_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    alpha: float = 0.98,
    use_article1_split: bool = False,
) -> pd.DataFrame:
    print("\n" + "=" * 50)
    print(f"Running experiment on {dataset_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print("Using Article 1 (70/30, no FS)" if use_article1_split else "Using Article 2 (60/40, BA/ICA FS)")
    print("=" * 50)

    # Normalize AFTER imputation, before split.
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    test_size = 0.3 if use_article1_split else 0.4
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    results: List[ExperimentResult] = []

    if use_article1_split:
        classifiers = build_article1_classifiers()
        for name, base in classifiers.items():
            print(f"  Processing classifier: {name}")
            clf = safe_fit(base, Xtr, ytr)
            if clf is None:
                continue
            ypred = clf.predict(Xte)
            yproba = safe_predict_proba(clf, Xte)
            metrics = compute_metrics(yte, ypred, yproba)
            all_idx = list(range(X.shape[1]))
            results.append(
                ExperimentResult(dataset_name, "None", name, 0.0, len(all_idx), all_idx, metrics)
            )
    else:
        classifiers = build_core_classifiers()
        ba = BAFeatureSelector(n_bats=12, n_iter=40, alpha=alpha)
        ica = ICAFeatureSelector(n_countries=20, n_imperialists=5, n_iter=40, alpha=alpha)
        for name, base in classifiers.items():
            print(f"  Processing classifier: {name}")

            # BA
            try:
                print("    Running BA selection...")
                ba_mask, ba_fit = ba.select(base, Xtr, ytr, cv)
                ba_mask = sanitize_mask(ba_mask, Xtr.shape[1])
                clf_ba = safe_fit(base, Xtr[:, ba_mask], ytr)
                if clf_ba is not None:
                    ypb = clf_ba.predict(Xte[:, ba_mask])
                    proba_b = safe_predict_proba(clf_ba, Xte[:, ba_mask])
                    met_b = compute_metrics(yte, ypb, proba_b)
                    ba_idx = list(np.flatnonzero(ba_mask))
                    results.append(
                        ExperimentResult(dataset_name, "BA", name, ba_fit, len(ba_idx), ba_idx, met_b)
                    )
                else:
                    print("    Skipping BA evaluation (fit failed).")
            except Exception as e:
                print(f"    BA step failed for {name}: {e}")

            # ICA
            try:
                print("    Running ICA selection...")
                ica_mask, ica_fit = ica.select(base, Xtr, ytr, cv)
                ica_mask = sanitize_mask(ica_mask, Xtr.shape[1])
                clf_ic = safe_fit(base, Xtr[:, ica_mask], ytr)
                if clf_ic is not None:
                    ypi = clf_ic.predict(Xte[:, ica_mask])
                    proba_i = safe_predict_proba(clf_ic, Xte[:, ica_mask])
                    met_i = compute_metrics(yte, ypi, proba_i)
                    ica_idx = list(np.flatnonzero(ica_mask))
                    results.append(
                        ExperimentResult(dataset_name, "ICA", name, ica_fit, len(ica_idx), ica_idx, met_i)
                    )
                else:
                    print("    Skipping ICA evaluation (fit failed).")
            except Exception as e:
                print(f"    ICA step failed for {name}: {e}")

    if not results:
        raise RuntimeError("No successful results for this dataset.")

    # For each classifier choose the best selector by Test Accuracy
    rows = []
    model_names = sorted({r.clf_name for r in results})
    for name in model_names:
        cand = [r for r in results if r.clf_name == name]
        best = max(cand, key=lambda r: r.test_metrics["Accuracy"])
        sel_names = [feature_names[i] for i in best.feature_indices]
        rows.append(
            {
                "Dataset": dataset_name,
                "Methodology": "Article_1" if use_article1_split else "Article_2",
                "Classifier": name,
                "Best_Selector": best.selector,
                "CV_Fitness": "-" if best.selector == "None" else f"{best.cv_fitness:.4f}",
                "Test_Accuracy": f"{best.test_metrics['Accuracy']:.4f}",
                "Sensitivity": f"{best.test_metrics['Recall/Sensitivity']:.4f}",
                "Specificity": f"{best.test_metrics['Specificity']:.4f}",
                "F1_Score": f"{best.test_metrics['F1']:.4f}",
                "Precision": f"{best.test_metrics['Precision']:.4f}",
                "N_Features": best.n_features,
                "Selected_Features": "; ".join(sel_names[:5]) + ("..." if len(sel_names) > 5 else ""),
            }
        )
    df = pd.DataFrame(rows).sort_values("Test_Accuracy", ascending=False).reset_index(drop=True)

    print("\n=== Results ===")
    print(
        df[
            [
                "Methodology",
                "Classifier",
                "Best_Selector",
                "Test_Accuracy",
                "Sensitivity",
                "Specificity",
                "F1_Score",
                "N_Features",
            ]
        ].to_string(index=False)
    )
    return df


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Feature Selection (BA/ICA) on CSV datasets")
    parser.add_argument(
        "--wdbc_csv", type=str, default=None, help="Path to WDBC CSV (Diagnosis column: M/B or 0/1)"
    )
    parser.add_argument(
        "--coimbra_csv", type=str, default=None, help="Path to Coimbra CSV (Class/Classification column: 1/2)"
    )
    parser.add_argument("--use_builtin", action="store_true", help="Use built-in datasets if CSV not provided")
    parser.add_argument("--alpha", type=float, default=0.98, help="Alpha for fitness (accuracy vs sparsity)")
    parser.add_argument(
        "--compare_articles", action="store_true", help="Also run Article 1 baseline (70/30, no FS)"
    )
    args = parser.parse_args()

    # Auto-detect your filenames if present
    if args.wdbc_csv is None and os.path.exists("breast_cancer_wisconsin.csv"):
        args.wdbc_csv = "breast_cancer_wisconsin.csv"
    if args.coimbra_csv is None and os.path.exists("breast_cancer_coimbra.csv"):
        args.coimbra_csv = "breast_cancer_coimbra.csv"

    all_results = []

    # ---- WDBC ----
    try:
        if args.wdbc_csv is not None:
            print(f"Loading WDBC from CSV: {args.wdbc_csv}")
            Xw, yw, fn_w = load_dataset_from_csv(args.wdbc_csv, "wdbc")
        elif args.use_builtin:
            print("Loading WDBC from built-in sklearn/UCI…")
            Xw, yw, fn_w = load_wdbc_builtin()
        else:
            raise ValueError(
                "WDBC not provided. Pass --wdbc_csv or place 'breast_cancer_wisconsin.csv' in this folder or add --use_builtin."
            )
        if args.compare_articles:
            all_results.append(run_experiment_on_dataset(Xw, yw, fn_w, "WDBC", args.alpha, use_article1_split=True))
        all_results.append(run_experiment_on_dataset(Xw, yw, fn_w, "WDBC", args.alpha, use_article1_split=False))
    except Exception as e:
        print(f"Error processing WDBC: {e}")

    # ---- COIMBRA ----
    try:
        if args.coimbra_csv is not None:
            print(f"Loading COIMBRA from CSV: {args.coimbra_csv}")
            Xc, yc, fn_c = load_dataset_from_csv(args.coimbra_csv, "coimbra")
        elif args.use_builtin:
            print("Loading COIMBRA from UCI…")
            Xc, yc, fn_c = load_coimbra_builtin()
        else:
            print("Skipping COIMBRA (no CSV and --use_builtin not set).")
            Xc = yc = fn_c = None
        if Xc is not None:
            all_results.append(
                run_experiment_on_dataset(Xc, yc, fn_c, "COIMBRA", args.alpha, use_article1_split=False)
            )
    except Exception as e:
        print(f"Error processing COIMBRA: {e}")

    # ---- Save combined results ----
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = "breast_cancer_feature_selection_results.csv"
        combined.to_csv(out_path, index=False)
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {out_path}")
    else:
        print("No datasets were successfully processed!")


if __name__ == "__main__":
    main()
