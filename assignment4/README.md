# Machine Learning I – Assignment 4
**Autoencoders and Image Restoration**

This repository contains the implementation of **Assignment 4** for the *Machine Learning I* course.

The assignment focuses on two applications of neural networks using the MNIST handwritten digits dataset:

- Learning a low-dimensional representation with a shallow autoencoder.
- Restoring partially occluded images through supervised reconstruction.

The experiments are carried out using TensorFlow/Keras and NumPy.

---

## Project Structure

```
.
├── src
│   ├── autoencoder.py
│   └── restore.py
└── README.md
```

> **Note:** The report is not included in this repository.

---

## Requirements

The project was developed with Python 3 and requires the following packages:

- tensorflow
- numpy
- matplotlib

They can be installed with:

```bash
pip install tensorflow numpy matplotlib
```

---

## Scripts

### `autoencoder.py`

Implements **Task 1** and **Task 2** of the assignment.

Main features:

- Downloads the MNIST dataset.
- Normalizes pixel values to the range `[0,1]`.
- Selects two digit classes (default: **5** and **7**).
- Trains a shallow autoencoder with architecture:

```
784 → 2 → 784
```

- Produces a two-dimensional latent embedding of the test images.
- Saves the embedding plot as:

```
embedding_5_vs_7.png
```

---

### `restore.py`

Implements **Task 3**.

Main features:

- Creates masked versions of MNIST images.
- Supports three masking strategies:
  - horizontal stripe
  - vertical stripe
  - rectangular patch
- Trains a reconstruction network to recover the original images.
- Displays masked, reconstructed, and original images.
- Saves the resulting figure according to the selected mask:

```
restoration_horizontal_5_vs_7.png
restoration_vertical_5_vs_7.png
restoration_rectangle_5_vs_7.png
```

---

## Running the Experiments

Run the scripts from the project root.

### Autoencoder

```bash
python src/autoencoder.py
```

### Image Restoration

```bash
python src/restore.py
```

The masking strategy can be selected by modifying:

```python
mask_type = "horizontal"
```

Available options are:

```python
"horizontal"
"vertical"
"rectangle"
```

---

## Dataset

The experiments use the **MNIST** handwritten digit dataset, automatically downloaded through `tensorflow.keras.datasets.mnist`.

Only two digit classes are used during training and evaluation (default: **5** and **7**).

---

## Notes

- Both scripts include a `DEBUG` flag that enables additional sanity checks and visualizations.
- The scripts automatically download the dataset if it is not already available.
- Random seeds are used for reproducibility of the masking procedure.