import time

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

X, Y = sklearn.datasets.fetch_california_housing(return_X_y=True)
Y = np.expand_dims(Y, axis=1)


def O_fun(x):
    return np.sqrt( np.sum( np.power( (x - np.mean(x)), 2 ) ) / len(x) )

def stand(x):
    return ( x - np.mean(x) ) / O_fun(x)

def normilize(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    return (x - (x_max + x_min) * 0.5) / ((x_max - x_min) * 0.5)

X = stand(X)
Y = stand(Y)

class Variable:
    def __init__(self, value):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.random((out_features, in_features))
        )
        self.b = Variable(
            value=np.zeros((out_features,))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.matmul(self.W.value, x.value.transpose()).transpose() + self.b.value
        )
        return self.output # this output will be input for next function in model

    def backward(self):
        # W*x + b / d b = 0 + b^{1-1} = 1
        # d_b = 1 * chain_rule_of_prev_d_func
        self.b.grad = 1 * self.output.grad

        # d_W = x * chain_rule_of_prev_d_func
        self.W.grad = np.matmul(
            np.expand_dims(self.output.grad, axis=2),
            np.expand_dims(self.x.grad, axis=1),
        )

        # d_x = W * chain_rule_of_prev_d_func
        self.x.grad = np.matmul(
            self.W.value.transpose(),
            self.output.grad.transpose()
        ).transpose()


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            (x.value >= 0) * x.value
        )
        return self.output

    def backward(self):
        self.x.grad = (self.x.value >= 0) * self.output.grad


class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            1.0 / (1.0 + np.exp(-x.value))
        )
        return self.output

    def backward(self):
        self.x.grad = self.output.value * (1.0 - self.output.value) * self.output.grad


class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean( np.power( (self.y.value - self.y_prim.value), 2 ) )
        return loss

    def backward(self):
        self.y_prim.grad = -2 * (self.y.value - self.y_prim.value)


class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad = (self.y_prim.value - self.y.value) / np.abs(self.y.value - self.y_prim.value)

class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=8, out_features=16),
            LayerSigmoid(),
            LayerLinear(in_features=16, out_features=4),
            LayerSigmoid(),
            LayerLinear(in_features=4, out_features=1)
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate


LEARNING_RATE = 1e-2
BATCH_SIZE = 16

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    LEARNING_RATE
)
loss_fn = LossMSE()

np.random.seed(0)
idxes_rand = np.random.permutation(len(X))
X = X[idxes_rand]
Y = Y[idxes_rand]

idx_split = int(len(X) * 0.8)
dataset_train = (X[:idx_split], Y[:idx_split])
dataset_test = (X[idx_split:], Y[idx_split:])

np.random.seed(int(time.time()))

losses_train = []
losses_test = []
for epoch in range(1, 301):

    for dataset in [dataset_train, dataset_test]:
        X, Y = dataset
        losses = []
        for idx in range(0, len(X)-BATCH_SIZE, BATCH_SIZE):
            x = X[idx:idx + BATCH_SIZE]
            y = Y[idx:idx + BATCH_SIZE]

            y_prim = model.forward(Variable(x))
            loss = loss_fn.forward(Variable(y), y_prim)

            losses.append(loss)

            if dataset == dataset_train:
                loss_fn.backward()
                model.backward()
                optimizer.step()

        if dataset == dataset_train:
            losses_train.append(np.mean(losses))
        else:
            losses_test.append(np.mean(losses))
    print(f'epoch: {epoch}     loss_train: {losses_train[-1]}     loss_test: {losses_test[-1]}')

    if epoch%30 == 0:
        plt.plot(losses_train)
        plt.plot(losses_test)
        plt.show()