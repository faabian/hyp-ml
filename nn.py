import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import zscore
from sklearn.utils import shuffle
import matplotlib.pyplot as pp


def read(filename):
    Xs = []
    Ys = []
    with open(filename, 'r') as file:
        file.seek(0, 2)           # jump to the end
        end_loc = file.tell()    # save the end location
        file.seek(0)             # jump back to the beginning
        file.readline()          # skip the status line
        while file.tell() != end_loc:
            try:
                # omit description strings
                xs = [float(x) for x in file.readline().split(' ')[1:]]
            except ValueError:
                break

            try:
                ys1 = [float(x) for x in file.readline().split(' ')[1:]]
            except ValueError:    # missing invariant trace field
                file.readline()
                continue

            try:
                ys2 = [float(x) for x in file.readline().split(' ')[1:]]
            except ValueError:    # missing trace field
                continue

            Xs.append(xs)
            Ys.append(ys1 + ys2)
        return Xs, Ys


# read the data
# Xs: geometric invariants
# Ys: arithmetic invariants
Xs, Ys = read('table_wo_res3.txt')

# omit possibly incomplete last datapoint
Xs, Ys = np.array(Xs[:-1]), np.array(Ys[:-1])
Xs, Ys = shuffle(Xs, Ys)
print(Xs.shape, Ys.shape)

num_arith_invs = Ys.shape[1] // 2
Ys = Ys[:, :num_arith_invs]    # drop the trace field
# Ys = Ys[: , (num_arith_invs+1):]   # drop the invariant trace field

VAR = 3   # target variable Ys[:, VAR] to model (22: class number)
# model by all X variables and all other Y variables
Rs = np.concatenate((Xs, Ys[:, :VAR], Ys[:, (VAR+1):]), axis=1)
# normalize input variables to mean=0, std=1
Rs = np.nan_to_num(zscore(Rs, axis=0))

Ys = Ys[:, VAR:(VAR+1)]
# show distribution of target variable
pp.hist(Ys)
pp.show()
# normalize target variable to mean=0, std=1
# (= minor information leak from validation data)
Ys = np.nan_to_num(zscore(Ys, axis=0))
# show distribution of normalized target variable
pp.hist(Ys)
pp.show()

# define the neural network
# (Activation layers are the nonlinearities)
model = tf.keras.Sequential([
    layers.Dense(4096),
    layers.Activation('relu'),
    layers.Dense(4096),
    layers.Activation('relu'),
    layers.Dense(Ys.shape[1]),
])


# The loss function of an estimator which always estimates zero
# (i.e. the mean for normalized variables).
def mean_estim(y_true, y_pred):
    return y_true ** 2


# use L2 error for regression tasks and report the mean estimator loss
# during training
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(lr=0.001),
              metrics=[mean_estim],
              )

# split data in training data and validation data, train on training
# data and report performance on the validation data
model.fit(Rs, Ys, batch_size=8, epochs=100, validation_split=0.2)
