"""
script_task2.py
----------------
Trains a shallow neural network on the UCI-CBM dataset using a simple
hold-out split. Data is normalized, and the model is trained multiple
times with different random initializations (multi-start) to avoid poor
local minima. The best MSE across the runs is reported together with
training-loss curves.
"""

from pandas import read_csv
import numpy as np
import tensorflow.keras as tf_keras
import matplotlib.pyplot as plt


# read the data
data = read_csv('../UCI-CBM/data.txt', sep='\s+').values #use space as separator, and convert into nunmpy array
c = 2
d = data.shape[1]-c
n = data.shape[0] #number of observation

# permute your data (must be i.i.d. anyway)
data = data[np.random.permutation(n), :] 

# set up algorithm parameters
kfolds =  5 #number of times that i extract test-set
split_frac = 1./kfolds # fraction of data that belongs to test-set
num_trials = 5 #number of times that i do restart of training phase
num_epochs = 15 #how many iterations for adam for each training-set

# set up model hyperparameters (shallow model --> only the number h of hidden units)
h = 12 # for not overfitting use "h" small 

# set up evaluation parameters
# (range of percentiles to be used in final reporting)
lperc, hperc = 25, 75 #for splitting

# prepare the model (the model is sequential, this is 2-layers model)
model = tf_keras.Sequential(name = 'my_network')
model.add(tf_keras.layers.Dense(h,name='Layer_1',activation='sigmoid',input_shape=(d,))) #every elements of the layer is connected to the elements of the next layer
model.add(tf_keras.layers.Dense(c,name='Layer_2',activation='linear')) # if "linear", the activation can be omitted

# === SIMPLE HOLD-OUT SPLIT FOR TASK 2 ===
n_test = int(n * split_frac)
X_test = data[:n_test, :d]
Y_test = data[:n_test, -c:]
X_train = data[n_test:, :d]
Y_train = data[n_test:, -c:]

# normalise your data
minXvals = X_train.min(axis=0)
Xranges = X_train.max(axis=0)-minXvals
for i in range(d):
  if Xranges[i] != 0:
    X_train[:,i] = (X_train[:,i] - minXvals[i]) / Xranges[i]
    X_test[:,i] = (X_test[:,i] - minXvals[i]) / Xranges[i]

minYvals = Y_train.min(axis=0)
Yranges = Y_train.max(axis=0)-minYvals

for i in range(c):
  if Yranges[i] != 0:
    Y_train[:,i] = (Y_train[:,i] - minYvals[i]) / Yranges[i]
    Y_test[:,i] = (Y_test[:,i] - minYvals[i]) / Yranges[i]

# save original architecture
model_architecture = tf_keras.models.clone_model(model)


# MULTI-START TRAINING LOOP (Task 2)
# store a list of histories for plotting at the end
histories = []
msevals = []
msebest = None

for trial in range(num_trials):
  print(f"--- trial no. {trial+1}/{num_trials} ---")

  # reset model to original architecture (untrained)
  model = tf_keras.models.clone_model(model_architecture)

  # compile model before training
  model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())

  # train and store history
  hist = model.fit(X_train, Y_train, batch_size=100, validation_split=.1, epochs=num_epochs)
  histories.append(hist.history)

  # evaluate on test (hold-out)
  Y_pred = model.predict(X_test)
  mse = ((Y_pred - Y_test)**2).sum()/(n_test*c)
  msevals.append(mse)

  # keep best mse
  if msebest is None or mse < msebest:
    msebest = mse

print(msevals)
print("")

# use collected values to compute median and percentiles
# the median will be the estimated performance
# the interval [25th, 75th] will be the range where the performance
# occurs half of the time
if len(msevals)>0:
  m, r, p = np.percentile(msevals,(hperc,50,lperc))
  print(f"mse = {r:.3E} (typical), mse in [{m:.3E}, {p:.3E}] with probability >= {(hperc-lperc)/100.:.2f}")

# === PLOT SOME LOSS HISTORIES ===
# pick best and worst trials
idx_best = int(np.argmin(msevals))
idx_worst = int(np.argmax(msevals))

plt.figure(figsize=(8,5))
plt.plot(histories[idx_best]['loss'], label='best run')
plt.plot(histories[idx_worst]['loss'], label='worst run')

plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss Histories (Task 2)')
plt.legend()
plt.grid(True)

# Save figure for report
filename = "task2_plot.png"
plt.savefig(filename, dpi = 200)
print(f"Saved plot to:  {filename}")

plt.show()