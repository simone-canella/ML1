"""
autoencoder.py
--------------
Assignment 4 — Task 1 & Task 2

Task 1: Load MNIST, normalize to [0, 1], and select two digit classes (parametric).
Task 2: Train a shallow autoencoder (784 → 2 → 784) with MSE reconstruction loss and
        plot the 2D embedding given by the hidden layer (encoder output).
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist


# -------------------------------------------------
# Debug flag: enable / disable optional checks
# -------------------------------------------------
DEBUG = True

# Load MNIST (images are 28x28, labels are digits 0..9)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] for stable training
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

if DEBUG:
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print("pixel range:", X_train.min(), X_train.max())

# Choose any two digits (make experiments easy by changing this tuple)
classes = (5, 7)
c0, c1 = classes

# Filter: keep only the two selected classes
train_mask = (y_train == c0) | (y_train == c1)
test_mask = (y_test == c0) | (y_test == c1)

X_train_2 = X_train[train_mask]
y_train_2 = y_train[train_mask]
X_test_2 = X_test[test_mask]
y_test_2 = y_test[test_mask]

if DEBUG:
    print("Filtered train:", X_train_2.shape, y_train_2.shape, "labels:", np.unique(y_train_2))
    print("Filtered test :", X_test_2.shape,  y_test_2.shape,  "labels:", np.unique(y_test_2))

# Flatten images to vectors so we can use a Dense network (784 = 28*28)
X_train_2 = X_train_2.reshape((-1, 784))
X_test_2 = X_test_2.reshape((-1, 784))


if DEBUG:
    print("Flattened train:", X_train_2.shape)
    print("Flattened test :", X_test_2.shape)

# -----------------------------------------
# Task 2: Shallow autoencoder (784 → 2 → 784)
# -----------------------------------------

input_dim = 784
encoded_dim = 2  # required by assignment to get a 2D embedding

# Build the network using the Functional API so we can easily extract the hidden layer output
layer_0 = Input(shape=(input_dim,), name="input")
layer_1 = Dense(
    encoded_dim,
    activation="sigmoid", 
    kernel_initializer=RandomUniform(minval=-0.7, maxval=0.7),
    name="encoded",
)(layer_0)
layer_2 = Dense(
    input_dim,
    activation="linear",  # regression output
    name="reconstructed",
)(layer_1)

# Two models:
# - encoder: maps input image -> 2D embedding
# - autoencoder: maps input image -> reconstructed image
encoder = Model(inputs=layer_0, outputs=layer_1, name="encoder")
autoencoder = Model(inputs=layer_0, outputs=layer_2, name="autoencoder")

autoencoder.summary()

# Train as a regression problem: target = input, loss = MSE
autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3), 
    loss="mse"
)

history = autoencoder.fit(
    X_train_2, X_train_2,
    epochs=20,
    batch_size=256,
    shuffle=True,
    verbose=1
)
    
print("Final train MSE:", history.history["loss"][-1])

# -----------------------------------------
# Inference: read the 2D hidden-layer output
# -----------------------------------------

Z_test = encoder.predict(X_test_2, verbose=0)  # (N, 2)

if DEBUG:
    print("Z_test shape:", Z_test.shape)

mask0 = (y_test_2 == c0)
mask1 = (y_test_2 == c1)

# -------------------------
# Plot + save embedding
# -------------------------
plt.figure()
plt.scatter(Z_test[mask0, 0], Z_test[mask0, 1], s=10, alpha=0.7, label=str(c0))
plt.scatter(Z_test[mask1, 0], Z_test[mask1, 1], s=10, alpha=0.7, label=str(c1))
plt.xlabel("z1")
plt.ylabel("z2")
plt.title(f"2D embedding (classes {c0} vs {c1})")
plt.legend()
plt.tight_layout()

# Save figure for report
filename = f"embedding_{c0}_vs_{c1}.png"
plt.savefig(filename, dpi=200)
print(f"Saved embedding plot to: {filename}")

plt.show()

