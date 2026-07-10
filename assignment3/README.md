# Machine Learning I - Assignment 3

This repository contains the Python implementation developed for **Assignment 3** of the *Machine Learning I* course.

The objective of the assignment is to implement and evaluate a shallow neural network for a regression problem using the UCI-CBM dataset. The work includes hold-out evaluation, k-fold cross-validation, multi-start training, and model selection based on the number of hidden neurons.

## Project Structure

```
.
├── src/
│   ├── script_task2.py
│   ├── script_task3.py
│   └── UCI-CBM/
│       └── data.txt
└── README.md
```

> **Note:** The report (`assignment3.pdf`) is not included in this workspace.

## Requirements

The scripts were developed with Python 3 and require the following packages:

- numpy
- pandas
- matplotlib
- tensorflow (Keras API)

Install the required dependencies with:

```bash
pip install numpy pandas matplotlib tensorflow
```

## Scripts

### `script_task2.py`

Performs hold-out evaluation of a shallow neural network.

Main steps:
- loads and normalizes the dataset;
- trains the network multiple times with different random initializations;
- evaluates the model on the hold-out test set;
- reports the MSE statistics;
- plots the training loss histories of the best and worst runs.

### `script_task3.py`

Performs model selection and final training.

Main steps:
- evaluates different numbers of hidden neurons using 5-fold cross-validation;
- applies multi-start training on each fold;
- selects the architecture with the lowest median MSE;
- retrains the best model on the complete dataset;
- plots the training and validation loss curves of the final model.

## Dataset

The scripts expect the UCI-CBM dataset to be located at:

```
src/UCI-CBM/data.txt
```

If the dataset is stored elsewhere, update the path inside the scripts accordingly.

## Running the Scripts

From the project root:

```bash
python src/script_task2.py
```

```bash
python src/script_task3.py
```

The scripts print the evaluation results to the console and display the corresponding plots.

## Author

**Simone Canella**  
Machine Learning I – University of Genoa