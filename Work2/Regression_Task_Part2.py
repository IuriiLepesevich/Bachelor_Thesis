import torch
import numpy as np
import threading

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W1 = 1
b1 = 1

W2 = 1
b2 = 1

a = 3

def linear(W, b, x):
    return (W * x) + b

def dW_linear(W, b, x):
    return (linear(W + 0.0001, b, x) - linear(W, b, x)) / 0.0001

def db_linear(W, b, x):
    return (linear(W, b + 0.0001, x) - linear(W, b, x)) / 0.0001

def dx_linear(W, b, x):
    return (linear(W, b, x + 0.0001) - linear(W, b, x)) / 0.0001



def tanh(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

def da_tanh(a):
    return (tanh(a + 0.0001) - tanh(a)) / 0.0001



def model(W2, b2, W1, b1, x, a):
    return LeakyReLU(linear(W2, b2, tanh(linear(W1, b1, x))), a)

def dW1_model(W2, b2, W1, b1, x, a):
    return d_LeakyReLU(linear(W2, b2, tanh(linear(W1, b1, x))), a) * W2 * da_tanh(linear(W1, b1, x)) * dW_linear(W1, b1, x)

def db1_model(W2, b2, W1, b1, x, a):
    return d_LeakyReLU(linear(W2, b2, tanh(linear(W1, b1, x))), a) * W2 * da_tanh(linear(W1, b1, x)) * db_linear(W1, b1, x)

def dW2_model(W2, b2, W1, b1, x, a):
    return d_LeakyReLU(linear(W2, b2, tanh(linear(W1, b1, x))), a) * tanh(linear(W1, b1, x))

def db2_model(W2, b2, W1, b1, x, a):
    return d_LeakyReLU(linear(W2, b2, tanh(linear(W1, b1, x))), a)



def LeakyReLU(z, a):
    if not isinstance(z, np.ndarray):
        z = np.array([z])
    ret_val = np.array(z)
    for i in range(np.prod(ret_val.shape)):
        if ret_val[i] <= 0:
            ret_val[i] *= a
    return ret_val


def d_LeakyReLU(z, a):
    temp = (LeakyReLU(z + 0.0001, a) - LeakyReLU(z, a)) / 0.0001
    return temp



def loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def dW1_loss(y, y_prim, W2, b2, W1, b1, x, a):
    return np.mean(-2 * (y - y_prim) * dW1_model(W2, b2, W1, b1, x, a))

def db1_loss(y, y_prim, W2, b2, W1, b1, x, a):
    return np.mean(-2 * (y - y_prim) * db1_model(W2, b2, W1, b1, x, a))

def dW2_loss(y, y_prim, W2, b2, W1, b1, x, a):
    return np.mean(-2 * (y - y_prim) * dW2_model(W2, b2, W1, b1, x, a))

def db2_loss(y, y_prim, W2, b2, W1, b1, x, a):
    return np.mean(-2 * (y - y_prim) * db2_model(W2, b2, W1, b1, x, a))

learning_rate = 1e-2

for epoch in range(15000):
    Y_prim = model(W2, b2, W1, b1, X, a)
    lossv = loss(Y, Y_prim)

    dW1_lossv = dW1_loss(Y, Y_prim, W2, b2, W1, b1, X, a)
    dW2_lossv = dW2_loss(Y, Y_prim, W2, b2, W1, b1, X, a)

    db1_lossv = db1_loss(Y, Y_prim, W2, b2, W1, b1, X, a)
    db2_lossv = db2_loss(Y, Y_prim, W2, b2, W1, b1, X, a)


    W1 = W1 - learning_rate * dW1_lossv
    W2 = W2 - learning_rate * dW2_lossv


    b1 = b1 - learning_rate * db1_lossv
    b2 = b2 - learning_rate * db2_lossv

    print(f'Iteration: {epoch},   Y_prim: {Y_prim},    loss: {lossv}')

print(f'Weight1: {W1},    Weight2: {W2},      Bias1: {b1},       Bias2: {b2}')

print(f'Floor 4 = {model(W2, b2, W1, b1, 4, a)}')