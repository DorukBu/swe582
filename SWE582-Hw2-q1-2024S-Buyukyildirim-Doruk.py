# NAME:       Doruk Büyükyıldırım
# STUDENT ID: 2022719033
# DATE:       25.05.2024
# SUMMARY:    SPRING 2024 SWE582 - SWE 582: Sp. Tp. Machine Learning for Data Analytic Homework #2 Question 1 Due May 26, 2024 by 23:59
#             This Python code does the following:
#             -> Load the MNIST dataset using sklearn
#             -> Filter the dataset for digits 2, 3, 8, and 9
#             -> Normalize the data
#             -> Split the dataset into training and test sets
#             -> Train a 4-class SVM using LinearSVC
#             -> Train a 4-class SVM using an RBF kernel
#             -> Use a smaller subset of the training data for hyperparameter tuning to shorten the process for the RBF kernel
#             -> Evaluate both models on the test set
#             -> Plot some of the support vectors
#             Acknowledgement: Generative AI is used to explore sklearn functions, its usage, and plotting.
#             Required Libraries: numpy, matplotlib, sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Filter the dataset for digits 2, 3, 8, and 9
digits = [2, 3, 8, 9]
mask = np.isin(y.astype(int), digits)
X, y = X[mask], y[mask].astype(int)

# Normalize the data
X = X / 255.0

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a 4-class SVM using LinearSVC
param_grid_linear_svc = {'C': [0.1, 1, 10, 100]}
grid_search_linear_svc = GridSearchCV(LinearSVC(), param_grid_linear_svc, cv=5, n_jobs=-1, verbose=3)
grid_search_linear_svc.fit(X_train, y_train)
best_linear_svc = grid_search_linear_svc.best_estimator_

train_accuracy_linear_svc = best_linear_svc.score(X_train, y_train)
test_accuracy_linear_svc = best_linear_svc.score(X_test, y_test)

print(f'Best Linear SVC Parameters: {grid_search_linear_svc.best_params_}')
print(f'Training Accuracy: {train_accuracy_linear_svc}')
print(f'Test Accuracy: {test_accuracy_linear_svc}')

# Use a smaller subset of the training data for hyperparameter tuning to shorten the process
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42, stratify=y_train)

# Train a 4-class SVM using an RBF kernel
rbf_svm = SVC(kernel='rbf', cache_size=1000)
param_dist = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
random_search = RandomizedSearchCV(rbf_svm, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=3)
random_search.fit(X_train_subset, y_train_subset)
best_rbf_svm = random_search.best_estimator_

# Train the best model on the full training dataset
best_rbf_svm.fit(X_train, y_train)

# Evaluate the model
train_accuracy_rbf = best_rbf_svm.score(X_train, y_train)
test_accuracy_rbf = best_rbf_svm.score(X_test, y_test)

print(f'Best RBF SVM Parameters: {random_search.best_params_}')
print(f'Training Accuracy: {train_accuracy_rbf}')
print(f'Test Accuracy: {test_accuracy_rbf}')

# Use the same dataset and model to create the support_vectors variable
support_vectors = best_rbf_svm.support_
support_vectors_indices = support_vectors

# Plot some of the support vectors
plt.figure(figsize=(10, 10))
for i, index in enumerate(support_vectors_indices[:16]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train.iloc[index].to_numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
