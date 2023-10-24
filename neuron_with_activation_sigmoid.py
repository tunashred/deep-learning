import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from keras import Sequential

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix
Y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix

pos = Y_train == 1
neg = Y_train == 0

# fig, ax = plt.subplots(1, 1, figsize=(4, 3))
# ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label="y=1")
# ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
#            edgecolors='blue', lw=2)
#
# ax.set_ylim(-0.08, 1.1)
# ax.set_ylabel('y', fontsize=12)
# ax.set_xlabel('x', fontsize=12)
# ax.set_title('one variable plot')
# ax.legend(fontsize=12)
# plt.show()

# Logistic neuron
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1, activation='sigmoid', name='L1')
    ]
)

# model.summary()

logistic_layer = model.get_layer('L1')
w, b = logistic_layer.get_weights()
# print(w, b)
# print(w.shape, b.shape)

set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

a1 = model.predict(X_train[0].reshape(1, 1))
print(a1)

