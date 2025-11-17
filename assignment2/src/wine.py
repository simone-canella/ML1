#!/usr/bin/env python3
"""
wine.py
-----------
Applies a custom K-Nearest Neighbors (kNN) classifier to the Wine dataset
from the UCI Machine Learning Repository. The script performs feature
normalization, splits the dataset into training and test sets, and evaluates
the classifier for several values of k. Accuracy results and other metrics 
(precision, recall, F1-score) are calculated. A PCA visualization highlights 
misclassified samples. This experiment supports Task 3 of Assignment 2.
"""

import numpy as np
from KNNClassifier import Knn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DEBUG = False

'''
---------------------
TASK 1: PREPARING DATA
---------------------
'''
# IMPORT DATA:
wine = load_wine()
X, y = wine.data, wine.target

# NORMALIZE FEATURES:
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# SPLIT DATA:
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42
)

'''
---------------------
TASK 2: RUN EXPERIMENTS
---------------------
'''
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

print("\nRESULTS OF KNN CLASSIFIER (Wine Dataset)\n")

for k in k_values:
    model = Knn(k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = model.test(y_test, y_pred)
    accuracies.append(acc)

    precision = precision_score(y_test, y_pred, average="macro")
    recall    = recall_score(y_test, y_pred, average="macro")
    f1        = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nk = {k}")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

# MEAN AND STANDARD DEVIATION
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print("\nSUMMARY STATISTICS")
print("------------------")
print("Accuracies    :", np.round(accuracies, 3))
print("Mean accuracy :", round(mean_acc, 3))
print("Std deviation :", round(std_acc, 3))

'''
---------------------
PCA VISUALIZATION OF ERRORS
---------------------
'''
# Reduce test data to 2 dimensions
pca = PCA(n_components=2)
pca.fit(X_train)
X_pca = pca.transform(X_test)

# Find misclassified points
incorrect_mask = (y_test != y_pred)
X_error = X_pca[incorrect_mask]

colors = ['red', 'green', 'blue']

plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    marker='x',
    c=y_test,
    cmap=ListedColormap(colors),
    s=30
)

# Misclassified points = black circles
plt.scatter(
    X_error[:, 0], X_error[:, 1],
    facecolors='none',
    edgecolors='black',
    s=120,
    linewidths=1.5
)

plt.title("Wine Dataset — PCA Visualization of Misclassified Samples")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
