# 1. Libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from tensorflow.keras.datasets import fashion_mnist
import joblib

# -------------------------------
# 2. Load Dataset

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(f"Training set: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test set: {x_test.shape}, Labels: {y_test.shape}")

# -------------------------------
# 3. Visualize Sample Data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

rng = np.random.default_rng()
chosen = []
for c in range(10):
    idxs = np.where(y_train == c)[0]
    chosen.append(rng.choice(idxs))

all_idx = np.arange(len(y_train))
remaining = np.setdiff1d(all_idx, np.array(chosen), assume_unique=False)
extra = rng.choice(remaining, size=2, replace=False)
sample_indices = np.array(chosen + extra.tolist())

# plot as 2x6 grid
plt.figure(figsize=(12, 4))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 6, i + 1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(class_names[y_train[idx]], fontsize=7)
    plt.axis('off')
plt.suptitle("Random Fashion-MNIST 12 Samples")
plt.show()
plt.show(block=True)

# -------------------------------
# 4. Preprocessing
# Flatten 28x28 → 784 and normalize pixel values (0–1)

x_train_flat = x_train.reshape(len(x_train), -1) / 255.0
x_test_flat = x_test.reshape(len(x_test), -1) / 255.0

# Train-validation split
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_flat, y_train, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train Logistic Regression Model

print("Training Logistic Regression")
model = LogisticRegression(
    max_iter=1000,
    solver='saga',
    penalty='l2',            # L2 regularization to prevent overfitting
    multi_class='multinomial'
)
model.fit(x_train_final, y_train_final)

# -------------------------------
# 6. Evaluate Model

y_pred = model.predict(x_val)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Confusion Matrix
conf = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
plt.show(block=True)

# Correct Predictions
correct_indices = np.where(y_val == y_pred)[0][:5]
plt.figure(figsize=(10, 2))
for i, idx in enumerate(correct_indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_val[idx].reshape(28, 28), cmap='gray')
    plt.title(f"✓ {class_names[y_pred[idx]]}", fontsize=8)
    plt.axis('off')
plt.suptitle("Correct Predictions")
plt.show()
plt.show(block=True)

# Misclassified Predictions
incorrect_indices = np.where(y_val != y_pred)[0][:5]
plt.figure(figsize=(10, 2))
for i, idx in enumerate(incorrect_indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_val[idx].reshape(28, 28), cmap='gray')
    plt.title(f"✗ {class_names[y_pred[idx]]}\nTrue: {class_names[y_val[idx]]}", fontsize=6)
    plt.axis('off')
plt.suptitle("Misclassified Examples")
plt.show()
plt.show(block=True)

# -------------------------------
# 7. Cross-Validation

print("Running 5-fold cross-validation ")
# Cross-validation helps estimate model performance on unseen data more reliably
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, x_train_final, y_train_final, cv=skf, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

# -------------------------------
# 8. Save Model + Predict Demo

joblib.dump(model, "fashion_model.pkl")
print("Model saved to fashion_model.pkl")

# Reload and predict on one test sample
reloaded_model = joblib.load("fashion_model.pkl")
sample = x_test_flat[0].reshape(1, -1)
prediction = reloaded_model.predict(sample)
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted (new data): {class_names[prediction[0]]}")
plt.axis("off")
plt.show()
plt.show(block=True)