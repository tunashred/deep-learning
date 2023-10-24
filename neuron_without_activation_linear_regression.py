import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]], dtype=np.float32) # size in 1000s square feet
Y_train = np.array([[300.0], [500.0]], dtype=np.float32) # price in 1000s dollars

fig, ax = plt.subplots(1, 1)
ax.scatter(X_train, Y_train, marker='x', c='r', label='Data Points')
ax.legend()
ax.set_ylabel('Price')
ax.set_xlabel('Size')
# plt.show()

linear_layer = tf.keras.layers.Dense(units=1, activation='linear', )
# there are no weights because they are not instantiated yet

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

w, b = linear_layer.get_weights()
print(w, b)

# weights are initialized with random values
set_w = np.array([[200]])
set_b = np.array([100])

linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)
alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
print(alin)

prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b

print(prediction_tf, prediction_np)

ax.plot(X_train, prediction_tf)
plt.show()