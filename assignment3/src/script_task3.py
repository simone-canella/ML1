"""
script_task3.py
----------------
Performs model selection for a shallow neural network by testing several
values of hidden units h using K-fold cross-validation. For each fold and
each h, the model is trained with multiple random restarts (multi-start)
and the best fold MSE is recorded. After selecting the best h, the script
re-trains a final model on the full dataset and plots the training curves.
"""

from pandas import read_csv
import numpy as np
import tensorflow.keras as tf_keras
import matplotlib.pyplot as plt

# read the data
data = read_csv('../UCI-CBM/data.txt', sep='\s+').values
c = 2
d = data.shape[1] - c
n = data.shape[0]

# permute data
data = data[np.random.permutation(n), :]

# algorithm parameters
kfolds = 5
num_trials = 5
num_epochs = 15

# hyperparameters to test
h_list = [4, 8, 12, 16]

# percentiles
lperc, hperc = 25, 75

results = []

# ============================================================
# === TASK 3: MODEL SELECTION (K-fold + Multi-start) =========
# ============================================================

for h in h_list:
    print(f"\n===== Testing h = {h} =====")

    # build architecture
    model = tf_keras.Sequential(name='my_network')
    model.add(tf_keras.layers.Dense(h, activation='sigmoid', input_shape=(d,), name='Layer_1'))
    model.add(tf_keras.layers.Dense(c, activation='linear', name='Layer_2'))

    # save untrained architecture
    model_architecture = tf_keras.models.clone_model(model)

    mse_folds = []

    # ===== K-FOLD LOOP =====
    for k in range(kfolds):
        print(f"--- Fold {k+1}/{kfolds} ---")

        # fold split
        idxmin = int(n * k / kfolds)
        idxtop = int(n * (k+1) / kfolds)
        n_test = idxtop - idxmin

        X_test = data[idxmin:idxtop, :d]
        Y_test = data[idxmin:idxtop, -c:]

        idxtrain = list(range(idxmin)) + list(range(idxtop, n))
        X_train = data[idxtrain, :d]
        Y_train = data[idxtrain, -c:]

        # ===== NORMALIZATION =====
        # X normalization
        minXvals = X_train.min(axis=0)
        Xranges = X_train.max(axis=0) - minXvals

        for i in range(d):
            if Xranges[i] != 0:
                X_train[:, i] = (X_train[:, i] - minXvals[i]) / Xranges[i]
                X_test[:,  i] = (X_test[:,  i] - minXvals[i]) / Xranges[i]

        # Y normalization
        minYvals = Y_train.min(axis=0)
        Yranges = Y_train.max(axis=0) - minYvals

        for i in range(c):
            if Yranges[i] != 0:
                Y_train[:, i] = (Y_train[:, i] - minYvals[i]) / Yranges[i]
                Y_test[:,  i] = (Y_test[:,  i] - minYvals[i]) / Yranges[i]

        # ===== MULTI-START LOOP =====
        msebest = None

        for trial in range(num_trials):

            model = tf_keras.models.clone_model(model_architecture)
            model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())

            # train
            model.fit(
                X_train, Y_train,
                batch_size=100,
                validation_split=.1,
                epochs=num_epochs,
                verbose=0
            )

            # evaluate
            Y_pred = model.predict(X_test, verbose=0)
            mse = ((Y_pred - Y_test)**2).sum() / (n_test * c)

            if msebest is None or mse < msebest:
                msebest = mse

        # store fold result
        mse_folds.append(msebest)

    # ===== SUMMARY FOR THIS h =====
    p25, p50, p75 = np.percentile(mse_folds, (lperc, 50, hperc))
    results.append([h, p50, p25, p75])

    print(f"h = {h}: median MSE = {p50:.3E}, range = [{p25:.3E}, {p75:.3E}]")


# ============================================================
# === MODEL SELECTION TABLE ==================================
# ============================================================

print("\n================ MODEL SELECTION RESULTS ================\n")
print("   h    |   median MSE   |    25th pct    |    75th pct")
print("---------------------------------------------------------")
for h, p50, p25, p75 in results:
    print(f"{h:5d} | {p50:12.3E} | {p25:12.3E} | {p75:12.3E}")

best_row = min(results, key=lambda x: x[1])
best_h = best_row[0]

print(f"\nBest model: h = {best_h}\n")


# ============================================================
# === FINAL TRAINING ON ALL DATA =====================
# ============================================================

print(f"\n===== Final Training Using Best h = {best_h} =====\n")

# build final model
model = tf_keras.Sequential(name='final_model')
model.add(tf_keras.layers.Dense(best_h, activation='sigmoid', input_shape=(d,), name='Layer_1'))
model.add(tf_keras.layers.Dense(c, activation='linear', name='Layer_2'))

model_architecture = tf_keras.models.clone_model(model)

# normalize ALL data
minXvals = data[:, :d].min(axis=0)
Xranges = data[:, :d].max(axis=0) - minXvals

X_all = data[:, :d].copy()
for i in range(d):
    if Xranges[i] != 0:
        X_all[:, i] = (X_all[:, i] - minXvals[i]) / Xranges[i]

minYvals = data[:, -c:].min(axis=0)
Yranges = data[:, -c:].max(axis=0) - minYvals

Y_all = data[:, -c:].copy()
for i in range(c):
    if Yranges[i] != 0:
        Y_all[:, i] = (Y_all[:, i] - minYvals[i]) / Yranges[i]

# final multi-start
best_final_loss = None
best_history = None

for trial in range(num_trials):
    print(f"> final trial {trial+1}/{num_trials}")

    model = tf_keras.models.clone_model(model_architecture)
    model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())

    hist = model.fit(
        X_all, Y_all,
        batch_size=100,
        validation_split=.1,
        epochs=num_epochs,
        verbose=0
    )

    final_loss = hist.history['loss'][-1]

    if best_final_loss is None or final_loss < best_final_loss:
        best_final_loss = final_loss
        best_history = hist

print(f"\nBest final training loss = {best_final_loss:.3E}")


# ============================================================
# === PLOT FINAL TRAINING CURVES =============================
# ============================================================

plt.figure(figsize=(8,5))
plt.plot(best_history.history['loss'], label='training loss')
plt.plot(best_history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Final Model Training (h = {best_h})')
plt.grid(True)
plt.legend()

# Save figure for report
filename = "task3_plot.png"
plt.savefig(filename, dpi = 200)
print(f"Saved plot to:  {filename}")

plt.show()
