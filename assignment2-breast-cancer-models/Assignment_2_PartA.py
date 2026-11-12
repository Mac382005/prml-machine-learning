# Breast Cancer Classification - Assignment 2 Part A (with Cross-Validation)
# Student Project: UCI Breast Cancer Wisconsin (Diagnostic)
# Author: Hoang Nhat Minh Mac

import warnings

import matplotlib.pyplot as plt
# 1. Imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# 2. Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Flip labels to match project goal: 1 = Malignant, 0 = Benign
# (sklearn original: 0 = malignant, 1 = benign)
y = 1 - y

print("Dataset loaded.")
print("Shape:", X.shape)
print("Target Distribution:\n", y.value_counts())

# 3. Exploratory Visualisations
# Class distribution
sns.countplot(x=y)
plt.title("Class Distribution (1 = Malignant, 0 = Benign)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# === Feature Importance (SelectKBest) Visualization ===
# Compute F-scores for all 30 features
selector_all = SelectKBest(score_func=f_classif, k='all')
selector_all.fit(X, y)
scores = selector_all.scores_

# Sort features by F-score (descending)
feat_scores = pd.DataFrame({
    'Feature': data.feature_names,
    'F_score': scores
}).sort_values(by='F_score', ascending=False).reset_index(drop=True)

# Plot top 15 features
plt.figure(figsize=(10, 6))
sns.barplot(x='F_score', y='Feature', data=feat_scores.head(15), palette='viridis')
plt.title("Top 15 Features by ANOVA F-Score (SelectKBest)")
plt.xlabel("F-Score (Feature Importance)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()
plt.savefig("feature_importance_top15.png", dpi=300)

# 4. Hold-out Preprocessing (Train/Test) for Final Spot Check
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection on the training set only to avoid leakage
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print("\nTop 15 Selected Features (Hold-out pipeline):")
print(selected_features.tolist())

# Data Preparation Summary
n_train = X_train_selected.shape[0]
n_test = X_test_selected.shape[0]
n_feats = X_train_selected.shape[1]

print("\n=== Data Preparation Summary ===")
print(f"Total features used: {n_feats} (k=15)")
print(f"Training samples: {n_train} (80% split)")
print(f"Testing samples:  {n_test} (20% split)")
print("Target encoding: 1 = Malignant, 0 = Benign")
print("Scaling method: StandardScaler (fit on train only)")

train_counts = pd.Series(y_train).value_counts().rename({0: "Benign (0)", 1: "Malignant (1)"})
test_counts  = pd.Series(y_test).value_counts().rename({0: "Benign (0)", 1: "Malignant (1)"})
print("\nClass distribution (train):")
print(train_counts)
print("\nClass distribution (test):")
print(test_counts)

# 5. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

# 6. Hold-out Evaluation (Confusion Matrix, Report, ROC)
def evaluate_model_holdout(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n=== HOLD-OUT: {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification Report:\n", classification_report(yte, y_pred))

    if y_prob is not None:
        auc = roc_auc_score(yte, y_prob)
        print(f"ROC AUC (hold-out): {auc:.4f}")
        fpr, tpr, _ = roc_curve(yte, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.figure(figsize=(8, 6))
for name, model in models.items():
    evaluate_model_holdout(name, model, X_train_selected, y_train, X_test_selected, y_test)

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves (Hold-out Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Cross-Validation (StratifiedKFold)
def cross_validate_models(models_dict, X_full, y_full, k_folds=5, k_features=15):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    rows = []
    for name, base_model in models_dict.items():
        # Build a pipeline: scale -> selectKBest -> model
        pipe = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=f_classif, k=k_features)),
            ('clf', base_model)
        ])

        # Compute CV metrics
        cv_acc = cross_val_score(pipe, X_full, y_full, cv=skf, scoring='accuracy')
        # ROC AUC requires probability estimates or decision function;
        # All chosen models support it via probability=True or built-in.
        cv_auc = cross_val_score(pipe, X_full, y_full, cv=skf, scoring='roc_auc')

        rows.append({
            'Model': name,
            'CV Folds': k_folds,
            'Accuracy Mean': np.mean(cv_acc),
            'Accuracy Std': np.std(cv_acc),
            'ROC AUC Mean': np.mean(cv_auc),
            'ROC AUC Std': np.std(cv_auc)
        })

    df_cv = pd.DataFrame(rows).sort_values(by='ROC AUC Mean', ascending=False).reset_index(drop=True)
    return df_cv

df_cv_summary = cross_validate_models(models, X, y, k_folds=5, k_features=15)
print("\n=== 5-Fold Cross-Validation Summary (Pipeline: Standardize -> SelectKBest(15) -> Model) ===")
print(df_cv_summary.to_string(index=False))

# Save CV summary to CSV
df_cv_summary.to_csv("cv_summary_results.csv", index=False)
print("\nSaved cross-validation summary to: cv_summary_results.csv")

# 8. Pick Best Model by CV AUC and Refit on Full Train
best_model_name = df_cv_summary.iloc[0]['Model']
print(f"\nBest model by CV ROC AUC: {best_model_name}")

# Recreate best pipeline and fit on the training split
best_base_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=15)),
    ('clf', best_base_model)
])
best_pipeline.fit(X_train, y_train)

# Evaluate best pipeline on the hold-out test set
y_pred_best = best_pipeline.predict(X_test)
y_prob_best = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None

print(f"\n=== HOLD-OUT EVALUATION OF BEST CV MODEL: {best_model_name} ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

if y_prob_best is not None:
    auc_best = roc_auc_score(y_test, y_prob_best)
    print(f"ROC AUC (hold-out, best CV model): {auc_best:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {auc_best:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve (Best CV Model on Hold-out Test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nDone.")
