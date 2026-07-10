"""
restore.py
----------
Assignment 4 — Task 3 (masked image restoration)

Goal:
Train a network to reconstruct the original MNIST image given a masked version.
This is a regression task on pixels, trained with MSE.
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------
# Debug flag: enable / disable optional checks
# -------------------------------------------------
DEBUG = True

# -------------------------
# Data preparation (as Task 1)
# -------------------------

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Choose two digit classes (parametric)
classes = (5, 7)
c0, c1 = classes

# Keep only the selected classes
train_mask = (y_train == c0) | (y_train == c1)
test_mask = (y_test == c0) | (y_test == c1)

X_train_2 = X_train[train_mask]
y_train_2 = y_train[train_mask]
X_test_2 = X_test[test_mask]
y_test_2 = y_test[test_mask]

if DEBUG:
    print("Train images:", X_train_2.shape)
    print("Test images :", X_test_2.shape)


# -------------------------
# Masking function
# -------------------------
def mask_horizontal_stripe(X, stripe_height=6, value=0.0, seed=0):
    """
    X: (N, 28, 28) in [0,1]
    returns a masked copy of X where one horizontal stripe is set to 'value'
    """
    rng = np.random.default_rng(seed)
    X_masked = X.copy()

    for i in range(X.shape[0]):
        r = rng.integers(0, 28 - stripe_height + 1)  # stripe start row
        X_masked[i, r:r + stripe_height, :] = value

    return X_masked

def mask_vertical_stripe(X, stripe_width=6, value=0.0, seed=0):
    """
    X: (N, 28, 28) in [0,1]
    returns a masked copy of X where one vertical stripe is set to 'value'
    """
    rng = np.random.default_rng(seed)
    X_masked = X.copy()

    for i in range(X.shape[0]):
        c = rng.integers(0, 28 - stripe_width + 1)  # stripe start column
        X_masked[i, :, c:c + stripe_width] = value

    return X_masked


def mask_rectangle_patch(X, patch_h=10, patch_w=10, value=0.0, seed=0):
    """
    X: (N, 28, 28) in [0,1]
    returns a masked copy of X where one rectangular patch is set to 'value'
    """
    rng = np.random.default_rng(seed)
    X_masked = X.copy()

    for i in range(X.shape[0]):
        r = rng.integers(0, 28 - patch_h + 1)  # top-left row
        c = rng.integers(0, 28 - patch_w + 1)  # top-left col
        X_masked[i, r:r + patch_h, c:c + patch_w] = value

    return X_masked

# -------------------------
# Create masked datasets (input = masked, target = original)
# -------------------------
mask_type = "rectangle"   # "horizontal", "vertical", "rectangle"

if mask_type == "horizontal":
    X_train_masked = mask_horizontal_stripe(X_train_2, stripe_height=6, value=0.0, seed=1)
    X_test_masked  = mask_horizontal_stripe(X_test_2,  stripe_height=6, value=0.0, seed=2)

elif mask_type == "vertical":
    X_train_masked = mask_vertical_stripe(X_train_2, stripe_width=6, value=0.0, seed=1)
    X_test_masked  = mask_vertical_stripe(X_test_2,  stripe_width=6, value=0.0, seed=2)

elif mask_type == "rectangle":
    X_train_masked = mask_rectangle_patch(X_train_2, patch_h=10, patch_w=10, value=0.0, seed=1)
    X_test_masked  = mask_rectangle_patch(X_test_2,  patch_h=10, patch_w=10, value=0.0, seed=2)

else:
    raise ValueError("Unknown mask_type. Use: 'horizontal', 'vertical', or 'rectangle'.")


if DEBUG:
    print("Masked train shape:", X_train_masked.shape)
    print("Example min/max:", X_train_masked[0].min(), X_train_masked[0].max())

# -------------------------
# Visual sanity check 
# -------------------------
if DEBUG:
    idx = 0  # pick one example

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(X_train_2[idx], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(X_train_masked[idx], cmap="gray")
    plt.title(f"Masked ({mask_type})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# Prepare inputs and targets
# -------------------------

# Flatten to 784-dimensional vectors for Dense network
X_train_in = X_train_masked.reshape((-1, 784))   # input: masked
X_train_out = X_train_2.reshape((-1, 784))       # target: original

X_test_in = X_test_masked.reshape((-1, 784))
X_test_out = X_test_2.reshape((-1, 784))

if DEBUG:
    print("Train input :", X_train_in.shape)
    print("Train target:", X_train_out.shape)

# -------------------------
# Reconstruction network
# -------------------------
inp = Input(shape=(784,), name="masked_input")
h1 = Dense(256, activation="relu", name="h1")(inp)
h2 = Dense(128, activation="relu", name="h2")(h1)
out = Dense(784, activation="linear", name="reconstruction")(h2)

model = Model(inp, out, name="restoration_mlp")
model.summary()

# -------------------------
# Training
# -------------------------
# Compile as regression on pixels
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

history = model.fit(
    X_train_in, X_train_out,
    validation_data=(X_test_in, X_test_out),
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1
)

print("Final val MSE:", history.history["val_loss"][-1])


# -------------------------
# Qualitative evaluation: show masked vs reconstructed vs original
# -------------------------
n_show = 5
pred = model.predict(X_test_in[:n_show], verbose=0)


# Reshape back to images
masked_imgs = X_test_in[:n_show].reshape((-1, 28, 28))
pred_imgs = pred.reshape((-1, 28, 28))
orig_imgs = X_test_out[:n_show].reshape((-1, 28, 28))

plt.figure(figsize=(10, 6))
for i in range(n_show):
    # Masked
    plt.subplot(3, n_show, i + 1)
    plt.imshow(masked_imgs[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.title("Masked")

    # Reconstructed (clip to keep values in visible range)
    plt.subplot(3, n_show, n_show + i + 1)
    plt.imshow(np.clip(pred_imgs[i], 0, 1), cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.title("Reconstructed")

    # Original
    plt.subplot(3, n_show, 2 * n_show + i + 1)
    plt.imshow(orig_imgs[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.title("Original")

plt.tight_layout()

# Save figure for report (optional)
filename = f"restoration_{mask_type}_{c0}_vs_{c1}.png"
plt.savefig(filename, dpi=200)
print(f"Saved reconstruction figure to: {filename}")

plt.show()
