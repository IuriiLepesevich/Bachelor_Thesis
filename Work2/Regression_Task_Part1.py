import torch
import numpy as np
import threading

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

def linear(W, b, x):
    return (W * x) + b

def dW_linear(W, b, x):
    return (linear(W + 0.0001, b, x) - linear(W, b, x)) / 0.0001

def db_linear(W, b, x):
    return (linear(W, b + 0.0001, x) - linear(W, b, x)) / 0.0001

def dx_linear(W, b, x):
    return (linear(W, b, x + 0.0001) - linear(W, b, x)) / 0.0001



def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def da_sigmoid(a):
    return (sigmoid(a + 0.0001) - sigmoid(a)) / 0.0001



def model(W, b, x):
    return sigmoid(linear(W, b, x)) * 20.0

def dW_model(W, b, x):
    return 20 * da_sigmoid(linear(W, b, x)) * dW_linear(W, b, x)

def db_model(W, b, x):
    return 20 * da_sigmoid(linear(W, b, x)) * db_linear(W, b, x)



def loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def dW_loss(y, y_prim, W, b, x):
    return np.mean(-2 * (y - y_prim) * dW_model(W, b, x))

def db_loss(y, y_prim, W, b, x):
    return np.mean(-2 * (y - y_prim) * db_model(W, b, x))

learning_rate = 1e-3

for epoch in range(5000):
    Y_prim = model(W, b, X)
    lossv = loss(Y, Y_prim)

    dW_lossv = dW_loss(Y, Y_prim, W, b, X)

    db_lossv = db_loss(Y, Y_prim, W, b, X)


    W = W - learning_rate * dW_lossv
    b = b - learning_rate * db_lossv

    print(f'Iteration: {epoch},   Y_prim: {Y_prim},    loss: {lossv}')

print(f'Weight: {W},    Bias: {b}')

print(f'Floor 4 = {model(W, b, 4)}')