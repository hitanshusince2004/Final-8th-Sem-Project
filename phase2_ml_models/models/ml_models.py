"""
Glacier Melting Detection — Machine Learning Models
====================================================
Linear Regression and Random Forest — regression & classification variants.

Each model class wraps sklearn with:
  - fit / predict / evaluate
  - feature importance (RF)
  - save / load
  - SHAP explainability (RF)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import seaborn as sns
from scipy import stats

from utils.metrics import regression_metrics, classification_metrics, print_metrics, save_metrics
from config.ml_config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    ML_FEATURE_COLS, REGRESSION_TARGET, BINARY_TARGET, MULTICLASS_TARGET,
    LR_PARAMS, LR_CLF_PARAMS, RF_PARAMS, RF_REGRESSION_PARAMS,
    RESULTS_DIR, CHECKPOINTS_DIR, N_CLASSES,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_tabular_data(split="train", target=REGRESSION_TARGET):
    """Load CSV split and return (X, y) numpy arrays."""
    path_map = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}
    df = pd.read_csv(path_map[split])
    available = [c for c in ML_FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df[target].values if target in df.columns else np.zeros(len(df))
    return X, y, available


# ══════════════════════════════════════════════════════════════════════════════
# BASE MODEL CLASS
# ══════════════════════════════════════════════════════════════════════════════

class GlacierMLModel:
    """Base class for all sklearn-based glacier models."""

    name        = "BaseModel"
    task        = "regression"   # or 'binary' or 'multiclass'
    target_col  = REGRESSION_TARGET

    def __init__(self):
        self.model        = None
        self.feature_names = None
        self.is_fitted    = False
        self.results      = {}

    def fit(self, X_train, y_train):
        print(f"\n[{self.name}] Training on {len(X_train):,} samples…")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"  Training complete")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X_test, y_test, split="test"):
        """Evaluate and return metrics dict."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        if self.task == "regression":
            metrics = regression_metrics(y_test, y_pred)
        else:
            avg = "binary" if self.task == "binary" else "macro"
            nc  = 2 if self.task == "binary" else N_CLASSES
            metrics = classification_metrics(y_test, y_pred, y_prob, average=avg, n_classes=nc)

        metrics["split"] = split
        metrics["model"] = self.name
        self.results[split] = metrics
        print_metrics(metrics, model_name=f"{self.name} [{split}]", task=self.task)
        return metrics

    def plot_distribution(self, X_test, y_test, save_path=None):
        """Plot the distribution of predicted vs actual values."""
        y_pred = self.predict(X_test)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(y_test, label="Actual", color="blue", alpha=0.5, kde=True, ax=ax)
        sns.histplot(y_pred, label="Predicted", color="orange", alpha=0.5, kde=True, ax=ax)
        ax.set_title(f"{self.name} — Distribution of Actual vs Predicted")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_residuals(self, X_test, y_test, save_path=None):
        """Plot residuals distribution and pattern."""
        y_pred = self.predict(X_test)
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual Distribution
        sns.histplot(residuals, kde=True, ax=ax1, color="red", alpha=0.6)
        ax1.axvline(0, color="black", linestyle="--")
        ax1.set_title(f"{self.name} — Residuals Distribution")
        ax1.set_xlabel("Residual (Actual - Predicted)")
        
        # Residual Pattern (Residuals vs Predicted)
        ax2.scatter(y_pred, residuals, alpha=0.4, s=10, color="purple")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title(f"{self.name} — Residual Pattern")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_predicted_vs_actual(self, X_test, y_test, save_path=None):
        """Scatter plot of predicted vs actual values."""
        y_pred = self.predict(X_test)
        metrics = self.results.get("test", {})

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10, color="#3B8BD4")
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lim, lim, "r--", linewidth=2, label="Perfect Fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{self.name} — Predicted vs Actual\n"
                     f"RMSE={metrics.get('RMSE', '?')}  R²={metrics.get('R2', '?')}")
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def save(self, path=None):
        path = path or os.path.join(CHECKPOINTS_DIR, f"{self.name.lower().replace(' ', '_')}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved → {path}")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_results(self, split="test"):
        path = os.path.join(RESULTS_DIR, f"{self.name.lower().replace(' ', '_')}_{split}_metrics.json")
        save_metrics(self.results.get(split, {}), path)


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR MODELS
# ══════════════════════════════════════════════════════════════════════════════

class GlacierLinearModel(GlacierMLModel):
    """Common base for linear regression variants."""
    def get_coefficients(self, feature_names=None):
        """Return DataFrame of feature coefficients (after scaling)."""
        # Find the linear part of the pipeline
        lin_model = None
        for step_name in ["ridge", "lasso", "elasticnet", "clf"]:
            if step_name in self.model.named_steps:
                lin_model = self.model.named_steps[step_name]
                break
        
        if lin_model is None:
            return None
            
        coef = lin_model.coef_
        if len(coef.shape) > 1: # For multiclass logistic
            coef = coef[0]
            
        names = feature_names or self.feature_names or [f"f{i}" for i in range(len(coef))]
        df = pd.DataFrame({
            "feature":     names,
            "coefficient": coef,
            "abs_coef":    np.abs(coef),
        }).sort_values("abs_coef", ascending=False)
        return df

    def plot_coefficients(self, feature_names=None, top_n=20, save_path=None):
        """Bar chart of top-N feature coefficients."""
        coef_df = self.get_coefficients(feature_names)
        if coef_df is None: return
        coef_df = coef_df[:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = ["#E8593C" if c > 0 else "#3B8BD4" for c in coef_df["coefficient"]]
        ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Coefficient value")
        ax.set_title(f"{self.name} — Top {top_n} coefficients")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig


class GlacierLinearRegressionStandard(GlacierLinearModel):
    """Standard Linear Regression."""
    name       = "Linear Regression"
    task       = "regression"
    target_col = REGRESSION_TARGET

    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression(fit_intercept=True)),
        ])

    def get_coefficients(self, feature_names=None):
        """Standard linear regression uses 'reg' step."""
        lin_model = self.model.named_steps["reg"]
        coef = lin_model.coef_
        names = feature_names or self.feature_names or [f"f{i}" for i in range(len(coef))]
        df = pd.DataFrame({
            "feature":     names,
            "coefficient": coef,
            "abs_coef":    np.abs(coef),
        }).sort_values("abs_coef", ascending=False)
        return df


class GlacierRidgeRegression(GlacierLinearModel):
    """Ridge Regression."""
    name       = "Ridge Regression"
    task       = "regression"
    target_col = REGRESSION_TARGET

    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self.alpha = alpha


class GlacierLassoRegression(GlacierLinearModel):
    """Lasso Regression (L1)."""
    name       = "Lasso Regression"
    task       = "regression"
    target_col = REGRESSION_TARGET

    def __init__(self, alpha=0.1):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso",  Lasso(alpha=alpha, fit_intercept=True)),
        ])
        self.alpha = alpha


class GlacierElasticNetRegression(GlacierLinearModel):
    """ElasticNet Regression (L1 + L2)."""
    name       = "ElasticNet Regression"
    task       = "regression"
    target_col = REGRESSION_TARGET

    def __init__(self, alpha=0.1, l1_ratio=0.5):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)),
        ])
        self.alpha = alpha


class GlacierLogisticRegression(GlacierLinearModel):
    """Logistic Regression."""
    name       = "Logistic Regression"
    task       = "binary"
    target_col = BINARY_TARGET

    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(**LR_CLF_PARAMS)),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

class GlacierRandomForestRegressor(GlacierMLModel):
    """Random Forest Regressor."""
    name       = "Random Forest Regressor"
    task       = "regression"
    target_col = REGRESSION_TARGET

    def __init__(self, params=None):
        super().__init__()
        p = {**RF_REGRESSION_PARAMS, **(params or {})}
        self.model = RandomForestRegressor(**p)

    def get_feature_importance(self, feature_names=None, top_n=30):
        names = feature_names or self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        df = pd.DataFrame({
            "feature":    names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return df

    def plot_feature_importance(self, feature_names=None, top_n=25, save_path=None):
        df = self.get_feature_importance(feature_names, top_n)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df["feature"], df["importance"], color="#1D9E75")
        ax.set_xlabel("Importance")
        ax.set_title(f"{self.name} — Top {top_n} Feature Importances")
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig


class GlacierExtraTreesRegressor(GlacierRandomForestRegressor):
    """Extra Trees Regressor."""
    name = "Extra Trees Regressor"
    def __init__(self, params=None):
        super().__init__(params)
        p = {**RF_REGRESSION_PARAMS, **(params or {})}
        if p.get("oob_score"):
            p["bootstrap"] = True
        self.model = ExtraTreesRegressor(**p)


class GlacierRandomForestClassifier(GlacierMLModel):
    """Random Forest Classifier."""
    name       = "Random Forest Classifier"
    task       = "binary"
    target_col = BINARY_TARGET

    def __init__(self, task="binary", params=None):
        super().__init__()
        self.task       = task
        self.target_col = BINARY_TARGET if task == "binary" else MULTICLASS_TARGET
        p = {**RF_PARAMS, **(params or {})}
        self.model = RandomForestClassifier(**p)

    def get_feature_importance(self, feature_names=None, top_n=30):
        names = feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        return pd.DataFrame({
            "feature":    names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

    def plot_feature_importance(self, feature_names=None, top_n=25, save_path=None):
        df = self.get_feature_importance(feature_names, top_n)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df["feature"], df["importance"], color="#534AB7")
        ax.set_xlabel("Importance")
        ax.set_title(f"{self.name} — Top {top_n} Feature Importances ({self.task})")
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_confusion_matrix(self, X_test, y_test, class_names=None, save_path=None):
        from sklearn.metrics import ConfusionMatrixDisplay
        y_pred = self.predict(X_test)
        classes_present = np.unique(np.concatenate([y_test, y_pred]))
        
        if class_names:
            names = [class_names[int(i)] for i in classes_present]
        else:
            default_names = ["Non-glacier", "Glacier"] if self.task == "binary" else ["Land", "Snow/Ice", "Water", "Debris"]
            names = [default_names[int(i)] for i in classes_present]
            
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=classes_present, display_labels=names,
            cmap="Blues", normalize="true", ax=ax
        )
        ax.set_title(f"{self.name} — Confusion Matrix ({self.task})")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_distribution(self, X_test, y_test, save_path=None):
        """Plot the distribution of predicted vs actual classes."""
        y_pred = self.predict(X_test)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = np.unique(np.concatenate([y_test, y_pred]))
        default_names = ["Non-glacier", "Glacier"] if self.task == "binary" else ["Land", "Snow/Ice", "Water", "Debris"]
        names = [default_names[int(i)] for i in classes]
        
        # Calculate counts for each class
        y_test_counts = [np.sum(y_test == c) for c in classes]
        y_pred_counts = [np.sum(y_pred == c) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, y_test_counts, width, label='Actual', color='blue', alpha=0.6)
        ax.bar(x + width/2, y_pred_counts, width, label='Predicted', color='orange', alpha=0.6)
        
        ax.set_ylabel('Count')
        ax.set_title(f'{self.name} — Class Distribution (Actual vs Predicted)')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig


class GlacierExtraTreesClassifier(GlacierRandomForestClassifier):
    """Extra Trees Classifier."""
    name = "Extra Trees Classifier"
    def __init__(self, task="binary", params=None):
        super().__init__(task, params)
        p = {**RF_PARAMS, **(params or {})}
        if p.get("oob_score"):
            p["bootstrap"] = True
        self.model = ExtraTreesClassifier(**p)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def train_ml_models(run_regression=True, run_classification=True):
    """
    Full training pipeline for all ML models.
    """
    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    all_results = {}

    # ── Regression Models ─────────────────────────────────────────────────────
    if run_regression:
        print("\n" + "═"*60)
        print("  PHASE 2a: Regression Models")
        print("═"*60)

        X_tr, y_tr, feats = load_tabular_data("train", REGRESSION_TARGET)
        X_vl, y_vl, _     = load_tabular_data("val",   REGRESSION_TARGET)
        X_te, y_te, _     = load_tabular_data("test",  REGRESSION_TARGET)

        regression_models = [
            GlacierLinearRegressionStandard(),
            GlacierRidgeRegression(alpha=1.0),
            GlacierLassoRegression(alpha=0.1),
            GlacierElasticNetRegression(alpha=0.1, l1_ratio=0.5),
            GlacierRandomForestRegressor(),
            GlacierExtraTreesRegressor(),
        ]

        for m in regression_models:
            print(f"\n--- Training {m.name} ---")
            m.feature_names = feats
            m.fit(X_tr, y_tr)
            m.evaluate(X_vl, y_vl, split="val")
            te_metrics = m.evaluate(X_te, y_te, split="test")
            
            # Save results
            m.save()
            m.save_results("test")
            all_results[m.name] = te_metrics

            # Comprehensive Plots
            base_name = m.name.lower().replace(" ", "_")
            m.plot_predicted_vs_actual(X_te, y_te, save_path=os.path.join(RESULTS_DIR, f"{base_name}_pred_vs_actual.png"))
            m.plot_distribution(X_te, y_te, save_path=os.path.join(RESULTS_DIR, f"{base_name}_dist.png"))
            m.plot_residuals(X_te, y_te, save_path=os.path.join(RESULTS_DIR, f"{base_name}_residuals.png"))
            
            if isinstance(m, GlacierLinearModel):
                m.plot_coefficients(feats, save_path=os.path.join(RESULTS_DIR, f"{base_name}_coef.png"))
            if isinstance(m, (GlacierRandomForestRegressor, GlacierExtraTreesRegressor)):
                m.plot_feature_importance(feats, save_path=os.path.join(RESULTS_DIR, f"{base_name}_feat_imp.png"))

    # ── Classification Models ─────────────────────────────────────────────────
    if run_classification:
        print("\n" + "═"*60)
        print("  PHASE 2b: Classification Models")
        print("═"*60)

        # Binary
        X_tr, y_tr, feats = load_tabular_data("train", BINARY_TARGET)
        X_vl, y_vl, _     = load_tabular_data("val",   BINARY_TARGET)
        X_te, y_te, _     = load_tabular_data("test",  BINARY_TARGET)

        binary_models = [
            GlacierLogisticRegression(),
            GlacierRandomForestClassifier(task="binary"),
            GlacierExtraTreesClassifier(task="binary"),
        ]

        for m in binary_models:
            print(f"\n--- Training {m.name} (Binary) ---")
            m.feature_names = feats
            m.fit(X_tr, y_tr)
            m.evaluate(X_vl, y_vl, split="val")
            te_metrics = m.evaluate(X_te, y_te, split="test")
            
            m.save()
            all_results[f"{m.name} (Binary)"] = te_metrics
            
            base_name = m.name.lower().replace(" ", "_") + "_binary"
            m.plot_distribution(X_te, y_te, save_path=os.path.join(RESULTS_DIR, f"{base_name}_dist.png"))
            
            if isinstance(m, GlacierLinearModel):
                m.plot_coefficients(feats, save_path=os.path.join(RESULTS_DIR, f"{base_name}_coef.png"))
            if isinstance(m, (GlacierRandomForestClassifier, GlacierExtraTreesClassifier)):
                m.plot_feature_importance(feats, save_path=os.path.join(RESULTS_DIR, f"{base_name}_feat_imp.png"))
                m.plot_confusion_matrix(X_te, y_te, save_path=os.path.join(RESULTS_DIR, f"{base_name}_cm.png"))

        # Multiclass
        X_tr_mc, y_tr_mc, _ = load_tabular_data("train", MULTICLASS_TARGET)
        X_vl_mc, y_vl_mc, _ = load_tabular_data("val",   MULTICLASS_TARGET)
        X_te_mc, y_te_mc, _ = load_tabular_data("test",  MULTICLASS_TARGET)

        multiclass_models = [
            GlacierRandomForestClassifier(task="multiclass"),
            GlacierExtraTreesClassifier(task="multiclass"),
        ]

        for m in multiclass_models:
            print(f"\n--- Training {m.name} (Multiclass) ---")
            m.feature_names = feats
            m.fit(X_tr_mc, y_tr_mc)
            m.evaluate(X_vl_mc, y_vl_mc, split="val")
            te_metrics = m.evaluate(X_te_mc, y_te_mc, split="test")
            
            m.save()
            all_results[f"{m.name} (Multiclass)"] = te_metrics
            
            base_name = m.name.lower().replace(" ", "_") + "_multiclass"
            m.plot_distribution(X_te_mc, y_te_mc, save_path=os.path.join(RESULTS_DIR, f"{base_name}_dist.png"))
            m.plot_feature_importance(feats, save_path=os.path.join(RESULTS_DIR, f"{base_name}_feat_imp.png"))
            m.plot_confusion_matrix(X_te_mc, y_te_mc, save_path=os.path.join(RESULTS_DIR, f"{base_name}_cm.png"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  ML MODEL COMPARISON SUMMARY")
    print("═"*60)
    for name, m in all_results.items():
        print(f"\n  {name}:")
        for k, v in m.items():
            if k not in ("split", "model"):
                print(f"    {k:<18} {v}")

    import json
    summary_path = os.path.join(RESULTS_DIR, "ml_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")

    return all_results


if __name__ == "__main__":
    train_ml_models()
