# Pattern Recognition and Machine Learning (PRML) – University of Canberra, 2025

Coursework from the PRML unit (Bachelor of Engineering (Honours), AI & Robotics), August–October 2025.
The main project is a breast cancer classifier on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

## Breast cancer detection (Assignments 2 and 3)

**Assignment 2** ([`assignment2-breast-cancer-models/`](assignment2-breast-cancer-models/)) compares four
classifiers with 5-fold stratified cross-validation:

| Model | CV accuracy | CV ROC AUC |
|---|---|---|
| Logistic Regression | 94.7% ± 1.8% | 0.992 |
| Random Forest | 94.6% ± 1.9% | 0.988 |
| SVM (RBF kernel) | 94.0% ± 2.0% | 0.991 |
| KNN (k = 5) | 93.5% ± 2.0% | 0.977 |

Also includes standalone SVM and decision tree scripts with confusion matrices, ROC curves and
feature importance plots.

**Assignment 3** ([`assignment3-breast-cancer-detection/`](assignment3-breast-cancer-detection/)) is the
full pipeline in [`breast_cancer_detection.py`](assignment3-breast-cancer-detection/breast_cancer_detection.py):

- hold-out and 5-fold CV evaluation of LR, SVM, RF and KNN
- GridSearchCV tuning of SVM and Random Forest, plus a soft-voting ensemble
- confusion matrices, ROC curves, learning curve and SelectKBest feature ranking
- a saved scikit-learn `Pipeline` (`final_model.joblib`) and demo predictions on new data

The tuned Random Forest scored 100% on the 114-case hold-out split. That split is small, so the
cross-validated figures above are the more reliable estimate. All outputs are in
[`ass3_outputs/`](assignment3-breast-cancer-detection/ass3_outputs/).

## Other work

- [`assignment1-fashion-mnist/`](assignment1-fashion-mnist/): logistic regression on Fashion-MNIST with stratified cross-validation
- [`labs/`](labs/): weekly labs on polynomial regression, MNIST logistic regression, regularisation and cross-validation, and Naive Bayes text classification (20 Newsgroups)

## Setup

```bash
pip install -r requirements.txt
python assignment3-breast-cancer-detection/breast_cancer_detection.py
```

All datasets load through scikit-learn or Keras, so nothing needs to be downloaded by hand.

Author: Hoang Nhat Minh Mac
