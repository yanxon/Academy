import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
print(m, n)

scaler = StandardScaler()
scaled_housing = scaler.fit_transform(housing.data)
scaled_housing_plus_bias = np.c_[np.ones((m,1)), scaled_housing]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_ = tf.matmul(X, theta, name="predictions")
error = y_ - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
