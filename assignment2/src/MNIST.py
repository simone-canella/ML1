#!/usr/bin/env python3
"""
mnist.py
-----------
Applies the custom K-Nearest Neighbors (kNN) classifier to the MNIST dataset.
The script loads the MNIST handwritten digit dataset, normalizes pixel values,
flattens each 28×28 image into a 784-dimensional vector, shuffles the dataset
and creates multiple training and testing subsets.

For each value of k, the classifier is trained on 6 training subsets of size
10,000 and evaluated on 5 testing subsets of size 2,000. The accuracy scores
collected across subsets are used to compute mean accuracy and standard
deviation, which are then plotted to analyze performance trends.
"""

import numpy as np
from KNNClassifier import Knn
from tensorflow.keras.datasets import mnist
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

DEBUG = False

'''
---------------------
TASK 1: PREPARING DATA
---------------------
'''

# IMPORT DATA:
(X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = mnist.load_data()

# NORMALIZE FEATURES:
X_train = X_train_orig.astype(np.float32) / 255.0
X_test  = X_test_orig.astype(np.float32)  / 255.0

# FLATTEN 28×28 IMAGES → vectors of dimension 784
N_train = X_train.shape[0]
N_test  = X_test.shape[0]
X_train = X_train.reshape((N_train, 784))
X_test  = X_test.reshape((N_test, 784))

# SHUFFLE
rng = np.random.default_rng(42)
perm_train = rng.permutation(N_train)
perm_test  = rng.permutation(N_test)

X_train = X_train[perm_train]
y_train = y_train_orig[perm_train]

X_test  = X_test[perm_test]
y_test  = y_test_orig[perm_test]

# CREATE SUBSETS 
train_subset_size = 10000
test_subset_size  = 2000

train_subsets = []
train_label_subsets = []

for i in range(6):
    start = i * train_subset_size
    end   = start + train_subset_size
    train_subsets.append(X_train[start:end])
    train_label_subsets.append(y_train[start:end])

test_subsets = []
test_label_subsets = []

for j in range(5):
    start = j * test_subset_size
    end   = start + test_subset_size
    test_subsets.append(X_test[start:end])
    test_label_subsets.append(y_test[start:end])

'''
---------------------
TASK 2: RUN EXPERIMENTS
---------------------
'''

k_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
mean_accuracy = []
std_accuracy = []

mean_precision = []
std_precision = []

mean_recall = []
std_recall = []

mean_f1 = []
std_f1 = []

print("MNIST kNN Experiment Results")
print("-----------------------------")

for k in k_values:
    if DEBUG:
        print("external for: ", k)
    
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(6):
        if DEBUG:
            print("  internal for: ", k,".", i)

        model = Knn(k)
        model.fit(train_subsets[i], train_label_subsets[i])
        
        for j in range(5):
            if DEBUG:
                print("    inner for: ", k,".", i, ".", j)
                
            y_pred = model.predict(test_subsets[j])

            acc = np.mean(y_pred == test_label_subsets[j])
            precision = precision_score(test_label_subsets[j], y_pred, average="macro")
            recall    = recall_score(test_label_subsets[j], y_pred, average="macro")
            f1        = f1_score(test_label_subsets[j], y_pred, average="macro")
            
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
    
    mean_accuracy.append(np.mean(acc_list))
    std_accuracy.append(np.std(acc_list))
    
    mean_precision.append(np.mean(precision_list))
    std_precision.append(np.std(precision_list))

    mean_recall.append(np.mean(recall_list))
    std_recall.append(np.std(recall_list))

    mean_f1.append(np.mean(f1_list))
    std_f1.append(np.std(f1_list))

    print(f"\nk = {k}")
    print(f"  Accuracy:  mean={mean_accuracy[-1]:.4f}, std={std_accuracy[-1]:.4f}")
    print(f"  Precision: mean={mean_precision[-1]:.4f}, std={std_precision[-1]:.4f}")
    print(f"  Recall:    mean={mean_recall[-1]:.4f}, std={std_recall[-1]:.4f}")
    print(f"  F1-score:  mean={mean_f1[-1]:.4f}, std={std_f1[-1]:.4f}")

'''
---------------------
TASK 3: PLOT RESULTS
---------------------
'''

plt.errorbar(k_values, mean_accuracy, yerr=std_accuracy, fmt='-o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('MNIST — kNN Accuracy vs. k (mean ± std)')
plt.grid(True)
plt.show()

'''
PCA VISUALIZATION (1000 samples)
'''
print("\nRunning PCA visualization on 1,000 sampled test images...")

sample_size = 1000
indices = np.random.choice(len(X_test), sample_size, replace=False)

X_sample = X_test[indices]
y_sample = y_test[indices]

best_k = k_values[np.argmax(mean_accuracy)]
model = Knn(best_k)
model.fit(X_train, y_train)
y_pred_sample = model.predict(X_sample)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

errors = X_pca[y_sample != y_pred_sample]

plt.figure(figsize=(8, 6))
cmap = ListedColormap([
    '#ff9999','#99ff99','#9999ff','#ffe699','#c2c2f0',
    '#ffb3e6','#ccff1a','#ffa64d','#c68c53','#80b3ff'
])

plt.scatter(X_pca[:, 0], X_pca[:, 1], 
            s=10, c=y_sample, cmap=cmap, alpha=0.6, marker='x')

plt.scatter(errors[:, 0], errors[:, 1],
            facecolors='none', edgecolors='black', s=80, linewidths=1.2)

plt.title(f'MNIST PCA (1,000 samples) — Misclassifications circled (k={best_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
