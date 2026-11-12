"""
Breast Cancer Classification - Assignment stage 3 (Final)
Student Project: UCI Breast Cancer Wisconsin (Diagnostic)
Author: Hoang Nhat Minh Mac

Stage 3 coverage:
- Evaluate algorithms (hold-out + 5-fold CV)
- Improve performance (GridSearchCV on SVM & RF)
- Soft-voting ensemble of tuned SVM+RF
- Present final results (metrics tables, confusion matrices, ROC)
- Visualise outcomes (confusion matrices, ROC curves, learning curve)
- Make predictions on new data (demo CSV)
- Create standalone final model (Pipeline)
- Save & reuse model (joblib)

Outputs are saved to: ./ass3_outputs  (relative to this .py file)
"""

import warnings

warnings.filterwarnings("ignore")

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ensure plots save to files (works in IDEs & headless)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

#  Paths 
RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(SCRIPT_DIR, "ass3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


#  Helpers
def safe_fname(s: str) -> str:
    r"""Windows-safe filename (remove :, /, \, etc.)."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Benign(0)", "Malignant(1)"], columns=["Pred 0", "Pred 1"])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Stage 2-like evaluation

def stage2_like_preprocess_and_eval():
    """
    Reproduce Stage 2 flow (scaler -> SelectKBest(k=15) -> models) and extend with saved plots/CSV.
    Also saves key exploratory plots to OUT_DIR:
      - class_distribution.png
      - feature_correlation_heatmap.png
      - top15_features_selectkbest.png
    Returns split data and selected features so later steps are consistent.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Flip labels so 1 = Malignant, 0 = Benign (sklearn original: 0=malignant, 1=benign)
    y = 1 - y

    #  EDA PLOTS (saved) 
    # 1) Class distribution
    plt.figure(figsize=(7.5, 5.5))
    sns.countplot(x=y)
    plt.title("Class Distribution (1 = Malignant, 0 = Benign)")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # 2) Feature correlation heatmap (30 features)
    plt.figure(figsize=(12, 10))
    corr = X.corr()  # ANOVA/linear corr visual only (not used for training)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 3) Top-15 features by ANOVA F-score (fit on full X,y for EDA ranking only)
    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X, y)
    f_scores = selector_all.scores_
    feat_rank = (
        pd.DataFrame({"Feature": data.feature_names, "F_score": f_scores})
        .sort_values("F_score", ascending=False)
        .reset_index(drop=True)
    )
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=feat_rank.head(15),
        x="F_score", y="Feature", palette="viridis"
    )
    plt.title("Top 15 Features by ANOVA F-Score (SelectKBest)")
    plt.xlabel("F-Score (higher is more discriminative)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top15_features_selectkbest.png"), dpi=300)
    plt.close()
    # -

    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Stage-2 style preprocessing on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    selected_features = X.columns[selector.get_support()]

    # Models (keep your Stage 2 set)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    # Hold-out ROC overlay + per-model confusion matrices & reports
    # Distinct colours/markers (longer cycles -> safe if you add models)
    color_cycle = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"])
    marker_cycle = cycle(["o", "s", "D", "^", "v", "<", ">", "p"])

    plt.figure(figsize=(8, 6))
    holdout_rows = []

    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        # Save confusion matrix + report
        save_confusion_matrix(
            y_test, y_pred,
            title=f"Confusion Matrix - {name} (hold-out)",
            path=os.path.join(OUT_DIR, f"confusion_matrix_holdout_{safe_fname(name)}.png")
        )
        with open(os.path.join(OUT_DIR, f"classification_report_holdout_{safe_fname(name)}.txt"), "w") as f:
            f.write(classification_report(y_test, y_pred, digits=4))

        row = {"Model": name, "Accuracy": float((y_pred == y_test).mean())}

        # Probabilities for ROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_sel)[:, 1]
            auc_val = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            color = next(color_cycle)
            marker = next(marker_cycle)
            # draw with small markers along the curve for differentiation
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2, color=color)
            # marker points every ~15 steps to avoid clutter
            step = max(1, len(fpr) // 15)
            plt.plot(fpr[::step], tpr[::step], linestyle="None", marker=marker, color=color, markersize=5)
            row["ROC AUC"] = float(auc_val)
        else:
            row["ROC AUC"] = np.nan

        holdout_rows.append(row)

    # Save overlay ROC for all models
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curves (Hold-out Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves_holdout_all.png"))
    plt.close()

    # Save holdout summary CSV
    df_holdout = pd.DataFrame(holdout_rows).sort_values("ROC AUC", ascending=False)
    df_holdout.to_csv(os.path.join(OUT_DIR, "holdout_summary_stage3.csv"), index=False)

    #  5-fold CV with full pipeline (scale -> selectKBest -> model) 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, base_model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=15)),
            ("clf", base_model),
        ])
        cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        cv_auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
        cv_rows.append({
            "Model": name,
            "CV Folds": 5,
            "Accuracy Mean": float(cv_acc.mean()),
            "Accuracy Std": float(cv_acc.std()),
            "ROC AUC Mean": float(cv_auc.mean()),
            "ROC AUC Std": float(cv_auc.std())
        })

    pd.DataFrame(cv_rows).sort_values("ROC AUC Mean", ascending=False) \
        .to_csv(os.path.join(OUT_DIR, "cv_summary_stage3.csv"), index=False)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "selected_features": list(selected_features)
    }


# Hyperparameter Tuning

def tune_top_two_models(X, y):
    """
    Hyperparameter tuning via GridSearchCV for SVM (RBF) and RandomForest
    inside a pipeline: scale -> selectKBest(k=15) -> clf.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # SVM grid
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])
    svm_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001]
    }
    svm_gs = GridSearchCV(svm_pipe, svm_grid, cv=skf, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0)
    svm_gs.fit(X, y)

    # RF grid
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=15)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt", "log2"]
    }
    rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=skf, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0)
    rf_gs.fit(X, y)

    # Save best params & CV scores
    results = {
        "svm": {"best_params": svm_gs.best_params_, "best_cv_auc": float(svm_gs.best_score_)},
        "rf": {"best_params": rf_gs.best_params_, "best_cv_auc": float(rf_gs.best_score_)}
    }
    with open(os.path.join(OUT_DIR, "gridsearch_best_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[GridSearch] SVM best:", svm_gs.best_params_, "CV AUC:", f"{svm_gs.best_score_:.4f}")
    print("[GridSearch]  RF best:", rf_gs.best_params_, "CV AUC:", f"{rf_gs.best_score_:.4f}")

    return svm_gs.best_estimator_, rf_gs.best_estimator_, results


# Best model evaluation

def evaluate_and_plot_best(name, pipeline, X_test, y_test, tag):
    """
    Evaluate a fitted pipeline on test set; save confusion matrix + ROC plot + report.
    Returns dict with accuracy and roc_auc (if available).
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Classification report
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    save_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix - {name}",
        path=os.path.join(OUT_DIR, f"confusion_matrix_{safe_fname(tag)}.png")
    )

    # ROC
    auc_val = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = float(auc(fpr, tpr))
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"roc_{safe_fname(tag)}.png"))
        plt.close()

    metrics = {
        "name": name,
        "accuracy": float((y_pred == y_test).mean()),
        "roc_auc": auc_val
    }
    with open(os.path.join(OUT_DIR, f"metrics_{safe_fname(tag)}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# Learning curve

def plot_learning_curve(estimator, X, y, title):
    """
    Plot learning curve (train vs validation accuracy across train sizes).
    Saves PNG to OUT_DIR (Windows-safe filename).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training", linewidth=2)
    plt.plot(train_sizes, val_mean, marker="s", label="Validation", linewidth=2)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = safe_fname(f"learning_curve_{title}.png")
    plt.savefig(os.path.join(OUT_DIR, fname))
    print("Saved learning curve:", os.path.join(OUT_DIR, fname))
    plt.close()


# Demo predictions & save

def demo_predictions(pipeline, X_test, n=5):
    """
    Demo predictions on n samples; saves CSV with predicted label and malignant probability.
    """
    n = min(n, len(X_test))
    probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else np.full(n, np.nan)
    preds = pipeline.predict(X_test)
    df = pd.DataFrame({
        "pred_label": preds[:n],
        "pred_prob_malignant": probs[:n]
    })
    df.to_csv(os.path.join(OUT_DIR, "demo_predictions.csv"), index=False)


def build_and_save_final_model(best_pipeline):
    """
    Save final pipeline as a reusable joblib object + label map + reuse README.
    """
    model_path = os.path.join(OUT_DIR, "final_model.joblib")
    joblib.dump(best_pipeline, model_path)

    with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
        json.dump({"0": "Benign", "1": "Malignant"}, f)

    with open(os.path.join(OUT_DIR, "README_REUSE_MODEL.txt"), "w") as f:
        f.write(
            "How to reuse the model:\n"
            "-\n"
            "import joblib\n"
            "pipe = joblib.load('final_model.joblib')\n"
            "y_pred = pipe.predict(X_new)\n"
            "y_prob = pipe.predict_proba(X_new)[:,1]\n"
        )
    return model_path


# Main

def main():
    print("[Stage 3] Reproducing & extending Stage 2 evaluation...")
    cache = stage2_like_preprocess_and_eval()
    X, y = cache["X"], cache["y"]
    X_train, X_test, y_train, y_test = cache["X_train"], cache["X_test"], cache["y_train"], cache["y_test"]

    print("[Stage 3] Hyperparameter tuning SVM & RF...")
    svm_best, rf_best, grid_results = tune_top_two_models(X, y)

    print("[Stage 3] Evaluating tuned SVM on hold-out...")
    svm_metrics = evaluate_and_plot_best("SVM (Tuned)", svm_best, X_test, y_test, tag="svm_tuned")

    print("[Stage 3] Evaluating tuned RF on hold-out...")
    rf_metrics = evaluate_and_plot_best("Random Forest (Tuned)", rf_best, X_test, y_test, tag="rf_tuned")

    print("[Stage 3] Building soft-voting ensemble...")
    ensemble = VotingClassifier(estimators=[("svm", svm_best), ("rf", rf_best)], voting="soft")
    ensemble.fit(X_train, y_train)
    ens_metrics = evaluate_and_plot_best("Soft Voting (SVM+RF tuned)", ensemble, X_test, y_test, tag="ensemble_soft")

    # Pick best by ROC AUC (fallback to accuracy if needed)
    candidates = [
        ("svm_tuned", svm_best, svm_metrics),
        ("rf_tuned", rf_best, rf_metrics),
        ("ensemble_soft", ensemble, ens_metrics)
    ]
    candidates.sort(key=lambda x: (x[2]["roc_auc"] if x[2]["roc_auc"] is not None else -1, x[2]["accuracy"]),
                    reverse=True)
    best_tag, best_pipeline, best_metrics = candidates[0]

    with open(os.path.join(OUT_DIR, "best_model_summary.json"), "w") as f:
        json.dump({"tag": best_tag, "metrics": best_metrics, "grid_cv": grid_results}, f, indent=2)

    print(f"[Stage 3] Plotting learning curve for best model: {best_tag}")
    # title kept human-readable; safe_fname() handles Windows filename
    plot_learning_curve(best_pipeline, X, y, title=f"Best Model: {best_tag}")

    print("[Stage 3] Making demo predictions...")
    demo_predictions(best_pipeline, X_test, n=5)

    print("[Stage 3] Saving final pipeline...")
    model_path = build_and_save_final_model(best_pipeline)

    summary_md = f"""# Assignment 3 Summary (Auto-generated)
- Date: {datetime.now().isoformat(timespec='seconds')}
- Best model tag: **{best_tag}**
- Holdout Accuracy: **{best_metrics['accuracy']:.4f}**
- Holdout ROC AUC: **{best_metrics['roc_auc'] if best_metrics['roc_auc'] is not None else 'N/A'}**
- Model saved to: `{model_path}`
- Plots & CSVs saved in: `{OUT_DIR}`
"""
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as f:
        f.write(summary_md)

    print("[Stage 3] Done. Outputs written to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
