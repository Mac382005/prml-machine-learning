## Week 3 Lab: MNIST Image Classification using Logistic Regression
# Objective¶
# By the end of this tutorial, you will:
# - Understand how to load and explore the MNIST dataset.
# - Preprocess images for logistic regression.
# - Train and evaluate a logistic regression classifier.
# - Interpret the results and identify strengths/weaknesses.

# Prerequisites
# You’ll need:
# - Python 3.x installed;
# - scikit-learn, numpy, matplotlib, and tensorflow installed.



## Step 1: Import libraries
# We import essential libraries for this task:
# - NumPy for numerical operations.
# - Matplotlib for data visualization.
# - scikit-learn for building and evaluating the logistic regression model. LogisticRegression from scikit-learn will be our classifier.
# - classification_report & confusion_matrix help evaluate performance.
# - TensorFlow to load the MNIST dataset easily. We use MNIST directly from Keras for convenience.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist


## Step 2: Load the MNIST dataset
# - MNIST contains 28×28 grayscale digit images (0–9).
# - _train has 60,000 images, x_test has 10,000.

# Load data: (x_train, y_train) for training, (x_test, y_test) for testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)



## Step 3: Visualize sample images
# This helps you see the images that your model will be classifying.
plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()



## Step 4: Preprocess the data
# - Flattening: Logistic regression expects 1D feature vectors.
# - Normalization: Helps model convergence and stability.

# Flatten each 28x28 image into a 784-length vector
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values to range [0, 1]
x_train_flat = x_train_flat.astype('float32') / 255.0
x_test_flat = x_test_flat.astype('float32') / 255.0



## Step 5: Build and Train the logistic regression model
# We create a multinomial logistic regression model using the 'saga' solver,
# suitable for multi-class problems and large datasets. Then, the model is
# trained on the flattened, normalized training images and their labels.
# - Multinomial logistic regression handles multiple classes.
# - saga solver works well for large datasets and supports multinomial loss.
# - During training, the model learns weights for each pixel to distinguish between digits.

# Using 'saga' solver for large datasets and multinomial classification
model = LogisticRegression(
    solver='saga',
    multi_class='multinomial',
    max_iter=100,      # Number of iterations
    verbose=1,         # Show training progress
    n_jobs=-1          # Use all CPU cores for speed
)

# Train the model
model.fit(x_train_flat, y_train)



## Step 6: Evaluate the model
# After training, we predict labels for the test dataset.
# - Accuracy gives an overall measure of correct predictions.
# - Confusion matrix shows per-class performance.
# - Classification report gives precision, recall, and F1 score for each digit.

# Accuracy on the test set
accuracy = model.score(x_test_flat, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions for detailed evaluation
y_pred = model.predict(x_test_flat)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))



## Step 7: Visualize some predictions
# Visualisation helps you better understand and interpret your results and model performance.
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.show()

# You can also visualize correct and incorrect predictioins separately.
# Find indices of correct and incorrect predictions
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

# Visualize first 10 correct predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(correct_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
    plt.axis('off')
plt.suptitle("Correct Predictions")
plt.show()

# Visualize first 10 incorrect predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}", color='red')
    plt.axis('off')
plt.suptitle("Incorrect Predictions")
plt.show()


## Expected Outcome
# Logistic regression will likely achieve ~92–93% accuracy on MNIST.
# Errors often occur between similar-looking digits (e.g., 4 vs. 9, 3 vs. 5).
# While logistic regression is simpler than deep neural networks, it’s still
# effective for image classification when the dataset is well-structured like MNIST.