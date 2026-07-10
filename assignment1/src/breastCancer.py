#!/usr/bin/env python3
"""
breastCancer.py
----------------
Implements and evaluates a Naive Bayes classifier on the UCI Breast Cancer dataset. 
The script loads and cleans categorical data, trains the custom NBayes model with Laplace smoothing, 
and evaluates performance using accuracy, precision, recall, and F1-score metrics. 
Results are visualized through PCA-based 2D projections highlighting misclassified samples.
"""

import pandas as pd
import numpy as np
from NBayesClassifier import Nbayes
from ucimlrepo import fetch_ucirepo

# import naive bayes classifier model
from sklearn.naive_bayes import GaussianNB
# import data loader
from sklearn.datasets import load_digits
# import performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import utility function to split the dataset
from sklearn.model_selection import train_test_split

# PLOT 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

'''
---------------------
TASK1: PREPARING DATA
---------------------
'''
#IMPORT DATA:
breast_cancer = fetch_ucirepo(id=14)

#EXTRACT FEATURES AND TARGETS
X = breast_cancer.data.features     # pandas DataFrame
y = breast_cancer.data.targets['Class']     # pandas DataFrame

#CLEAR DATA (substitute or delete missing values/row denoted by "?")
x = X.replace('?', 'Unknown')

#SPLIT DATA:
trainRatio = 0.8 #select the percentage of training-set

total_lenght = len(x)
train_lenght = int(total_lenght * trainRatio)

# Shuffle the indices
indices = np.arange(total_lenght) #create a list of all row indices
np.random.seed(42)  
np.random.shuffle(indices) #random rearrangement of the indices

# Split indices
train_indices = indices[:train_lenght]
test_indices = indices[train_lenght:]

# Create training-set and test-set
x_train = x.iloc[train_indices]
x_test = x.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

'''
---------------------
TASK2.1: FIT METHOD
---------------------
'''

model = Nbayes()

model.fit(x_train, y_train)

'''
---------------------
TASK2.2: PREDICT METHOD
---------------------
'''

y_pred = model.predict(x_test)

'''
---------------------
TASK2.3: TEST METHOD
---------------------
'''
accuracy = model.test(x_test, y_test)

print("RESULT OF NAYVE BAYES CLASSIFIER\n")
print("Accuracy         :", accuracy)

'''
---------------------
EVALUATE DATA 
---------------------
'''

# Evaluating the model
accuracy_sklearn = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print the results
print("Accuracy sklearn :", accuracy_sklearn)
print("Precision        :", precision)
print("Recall           :", recall)
print("F1 Score         :", f1)

'''
---------------------
VISUALIZE DATA
---------------------
'''
x_train_encoded = pd.get_dummies(x_train)
x_test_encoded = pd.get_dummies(x_test)
x_test_encoded = x_test_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)

# Fit PCA (reduce to 2D for visualization)
pca = PCA(n_components=2)
pca.fit(x_train_encoded)
X_pca = pca.transform(x_test_encoded)

# Encode target labels (y_test and y_pred) into numeric values for plotting 
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
y_pred_encoded = le.transform(y_pred)
X_error = X_pca[y_test_encoded != y_pred_encoded, :]

# Plot
colors = ['red','green','blue','cyan','magenta','yellow','lightblue','gray']

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=50, marker='x',
            c=y_test_encoded, cmap=ListedColormap(colors[:len(set(y_test_encoded))]),
            label='Samples')
plt.plot(X_error[:,0], X_error[:,1], 'ok', markersize=15,
         fillstyle='none', label='Misclassified')
plt.title("Naive Bayes Classification - PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure for report
filename = "breastCancer_plot.png"
plt.savefig(filename, dpi = 200)
print(f"Saved plot to:  {filename}")

plt.show()


