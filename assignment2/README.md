# Assignment 2 - K-Nearest Neighbors Classifier

This project was developed for the **Machine Learning I** course and consists of a complete implementation of the **k-Nearest Neighbors (kNN)** classification algorithm from scratch using Python.

The objective of the assignment is to implement a custom kNN classifier, evaluate it on two different datasets (Wine and MNIST), and analyze its performance for different values of *k*.

---

## Project Structure

```
assignment2/
│
├── src/
│   ├── KNNClassifier.py
│   ├── wine.py
│   └── mnist.py
│
└── README.md
```

> **Note:** The assignment report is stored outside this project folder.

---

## Files Description

### `KNNClassifier.py`

Implements the custom k-Nearest Neighbors classifier.

The `Knn` class provides three main methods:

- `fit(X_train, y_train)`
  - Stores the training dataset.

- `predict(X_test)`
  - Computes the Euclidean distance between each test sample and every training sample.
  - Selects the *k* nearest neighbors.
  - Predicts the class using majority voting.

- `test(y_test, y_pred)`
  - Computes the classification accuracy.

---

### `wine.py`

Runs experiments on the Wine dataset.

Main operations:

- Loads the Wine dataset.
- Normalizes all features into the interval `[0,1]`.
- Splits the dataset into training and testing sets.
- Evaluates the classifier for several values of *k*.
- Computes:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Computes the mean accuracy and standard deviation.
- Generates a PCA visualization highlighting misclassified samples.

---

### `mnist.py`

Runs experiments on the MNIST handwritten digit dataset.

Main operations:

- Loads the MNIST dataset.
- Normalizes pixel values.
- Flattens each 28×28 image into a 784-dimensional feature vector.
- Randomly shuffles the dataset.
- Creates:
  - 6 training subsets of 10,000 images
  - 5 testing subsets of 2,000 images
- Evaluates multiple values of *k*.
- Computes:
  - Mean accuracy
  - Standard deviation
  - Precision
  - Recall
  - F1-score
- Produces:
  - Accuracy vs. *k* plot
  - PCA visualization of 1,000 randomly selected test samples.

---

## Datasets

### Wine Dataset

- 178 samples
- 13 continuous features
- 3 wine classes

The dataset is provided by **scikit-learn**.

---

### MNIST Dataset

- 70,000 grayscale handwritten digit images
- Image size: 28 × 28 pixels
- 10 classes (digits 0–9)

The dataset is loaded using **TensorFlow/Keras**.

---

## Requirements

The project requires Python 3 together with the following libraries:

- numpy
- scipy
- matplotlib
- scikit-learn
- tensorflow

They can be installed with:

```bash
pip install numpy scipy matplotlib scikit-learn tensorflow
```

---

## Running the Project

Move into the `src` directory:

```bash
cd src
```

Run the Wine experiment:

```bash
python3 wine.py
```

Run the MNIST experiment:

```bash
python3 mnist.py
```

---

## Notes

The MNIST experiment is computationally demanding because kNN computes the distance between every test sample and every training sample.

On a standard CPU, the complete experiment may require **more than two hours** to finish.

The current implementation is entirely sequential and uses a single CPU core. Performance could be significantly improved through parallel processing or GPU acceleration.

---

## Author

**Simone Canella**

Machine Learning I  
University of Genoa