"""
KNNClassifier.py
--------------------
Implements a K-Nearest Neighbors classifier for data. 
Includes methods for training (`fit`), predicting new samples (`predict`), 
and evaluating model accuracy (`test`). 
The goal is to build a simple, interpretable classifier from scratch following ML theory.
"""

import numpy as np
from scipy.stats import mode

class Knn:
    """
    Initialize the kNN classifier.

    Parameters:
        k (int): Number of nearest neighbors to consider.
    """
    def __init__(self, k):
        self.k = k

        self.X_train = None
        self.y_train = None


        self.trained = False
        self.DEBUG = False


    """
    Store the training data.

    Parameters:
        X_train : Training features.
        y_train : Training labels.

    Return:
        None
    
    Raises:
        ValueError: If input shapes are inconsistent.
    """
    def fit(self, X_train, y_train):
        
        if len(X_train) != len(y_train):
          raise ValueError("X_train and y_train must have same number of samples.")

        self.X_train = X_train
        self.y_train = y_train
        self.n_train = len(y_train)

        self.trained = True

    """
    Predict the class labels for the test set.

    Parameters:
        X_test : Test features

    Returns:
        y_predict : Vector of predicted labels of shape
    
    Raises:
        ValueError: If the model has not been fitted yet.
    
    """
    def predict(self, X_test):
        if not self.trained:
            raise ValueError
        
        y_predict = [] # list of predicted labels based on majority vote among the k nearest neighbors

        for x in X_test:
            # Compute Euclidean distance from this test sample to all training samples
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Sort and pick k nearest indices
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            # Take the most common label
            predicted_label = mode(k_labels, keepdims = True)[0][0] #keepdims ensure the dimension

            y_predict.append(predicted_label)


        return np.array(y_predict)

    """
    Compute the classification accuracy.

    Parameters:
        y_test: True labels for the test set.
        y_pred: Predicted labels from predict().

    Returns:
        float: Accuracy score (fraction of correct predictions).
    
    Raises:
        ValueError: If the model has not been fitted yet.
    """
    def test(self, y_test, y_pred):
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        return (np.array(y_test) == np.array(y_pred)).sum() / len(y_test)


    