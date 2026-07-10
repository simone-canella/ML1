# Machine Learning - Lab 1: Naive Bayes Classifier

This project contains the implementation of a Naive Bayes classifier developed from scratch for the first Machine Learning laboratory assignment.

The objective is to build a classifier for categorical data without relying on machine learning libraries for the learning algorithm itself. The implementation follows the specifications provided during the course and includes Laplace (additive) smoothing to avoid zero-probability issues.

## Project structure

```
.
├── data/
│   └── weather.data.csv
├── src/
│   ├── NBayesClassifier.py
│   ├── weather.py
│   ├── breastCancer.py
│   ├── weather_plot.png
│   └── breastCancer_plot.png
└── README.md
```

## Requirements

The project was developed using Python 3.

Required packages:

- numpy
- pandas
- matplotlib
- scikit-learn
- ucimlrepo

They can be installed with:

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

## Datasets

Two datasets are used in this assignment.

### Weather dataset

A small categorical dataset (14 samples) used to verify the correctness of the implementation. Since the dataset is very small, the same data are used for both training and testing as a sanity check.

### Breast Cancer dataset

The Breast Cancer dataset is downloaded from the UCI Machine Learning Repository through the `ucimlrepo` package.

Before splitting the data, the samples are randomly shuffled because the original dataset is ordered by class. This avoids introducing bias into the training and test sets.

Missing values represented by `?` are replaced with the categorical value `"Unknown"`.

## Implementation

The classifier is implemented in the `NBayes` class and provides three methods:

- `fit(X_train, y_train)`  
  Computes the class priors and conditional probabilities.

- `predict(X_test)`  
  Predicts the class label for each test sample using the Naive Bayes decision rule.

- `test(X_test, y_test)`  
  Computes the classification accuracy.

Conditional probabilities are estimated using Laplace (additive) smoothing with smoothing parameter:

```
a = 1
```

## Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

For comparison, the same metrics are computed using the equivalent implementation available in scikit-learn.

## Visualization

Both scripts generate a two-dimensional visualization of the data using Principal Component Analysis (PCA).

Categorical variables are first one-hot encoded and then projected onto two principal components. Misclassified samples are highlighted with black circles.

The PCA plots are intended only for visualization and are not used during training or prediction.

## Running the code

Run the Weather example:

```bash
python src/weather.py
```

Run the Breast Cancer example:

```bash
python src/breastCancer.py
```

Both scripts print the evaluation metrics and display the PCA visualization.

Both scripts print the evaluation metrics and display the PCA visualization.

## Results

### Weather dataset

| Metric | Value |
|--------|-------:|
| Accuracy | 0.9286 |
| Precision | 0.9500 |
| Recall | 0.9000 |
| F1 Score | 0.9181 |

### Breast Cancer dataset

| Metric | Value |
|--------|-------:|
| Accuracy | 0.7759 |
| Precision | 0.7098 |
| Recall | 0.7186 |
| F1 Score | 0.7139 |

The custom implementation produces the same accuracy as the corresponding scikit-learn implementation on both datasets.

## Notes

This project was developed as part of the Machine Learning laboratory activities.

The implementation focuses on categorical Naive Bayes classification and follows the requirements of the assignment, including:

- categorical data preprocessing;
- Laplace smoothing;
- custom implementation of the learning algorithm;
- comparison with scikit-learn;
- visualization of classification results using PCA.