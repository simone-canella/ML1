#!/usr/bin/env python3
import pandas as pd
from NBayesClassifier import Nbayes

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

DEBUG = False

'''
---------------------
TASK1: PREPARING DATA
---------------------
'''
#IMPORT DATA:
df_weather = pd.read_csv("../data/weather.data.csv", sep='\s+')

if DEBUG == True:
    print(df_weather.shape)


#CLEAN DATA:
df_weather.columns = df_weather.columns.str.replace('#', '') #substitute '#' with '' from index

if DEBUG == True:
    print("UPLOADED DATAFRAME: \n", df_weather.columns, "\n") #control that is done correctly


#SPLIT DATA:
x_train = df_weather.iloc[:, 0 : -1] #select first 4 columns 

if DEBUG == True:
    print("TRAIN CONDITION: \n", x_train, "\n")

y_train = df_weather.iloc[:, -1] #select last column

if DEBUG == True:
    print("TRAIN EFFECT: \n", y_train, "\n")


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

y_pred = model.predict(x_train)

'''
---------------------
TASK2.3: TEST METHOD
---------------------
'''
accuracy = model.test(x_train, y_train)
print("RESULT OF NAYVE BAYES CLASSIFIER\n")
print("Accuracy         :", accuracy)

'''
---------------------
EVALUATE DATA AND VISUALIZE DATA
---------------------
'''
y_test = y_train
x_test = x_train

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

# Plot result

# Encode categorical features for PCA 
x_train_encoded = pd.get_dummies(x_train)

# Fit PCA (reduce to 2D for visualization)
pca = PCA(n_components=2)
pca.fit(x_train_encoded)
X_pca = pca.transform(x_train_encoded)

# Encode target labels (y_test and y_pred) into numeric values for plotting 
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
y_pred_encoded = le.transform(y_pred)
X_error = X_pca[y_test_encoded != y_pred_encoded, :]

# Define colors for each class 
colors = ['red','green','blue','cyan','magenta','yellow','lightblue','gray']

# Scatter plot of PCA projection 
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50, marker='x', c=y_test_encoded, cmap=ListedColormap(colors[:len(set(y_test_encoded))]), label='Samples')

# Mark the misclassified points 

plt.plot(X_error[:,0], X_error[:,1], 'ok', markersize=15, fillstyle='none', label='Misclassified')
plt.title("Naive Bayes Classification - PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

