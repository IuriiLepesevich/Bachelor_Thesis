import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F



plt.rcParams["figure.figsize"] = (12, 14) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


def O_fun(x):
    return np.sqrt( np.sum( np.power( (x - np.mean(x, axis=0)), 2 ), axis=0 ) / len(x) )

def stand(x):
    return ( x - np.mean(x, axis=0) ) / (O_fun(x) + 1e-8)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        X = np.array(pd.read_csv(".\For_modeling_temp.csv"))

        self.X = np.array(X[:, 2:]).astype(np.float32)
        X_std = stand(self.X)
        self.X = torch.FloatTensor(X_std)

        self.Y = np.array(X[:, 1]).astype(np.float32)
        Y_std = stand(self.Y)
        self.Y = torch.FloatTensor(Y_std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.expand_dims(self.Y[idx], axis=-1)


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class BatchNorm(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.x = None
        self.meanvar = None
        self.variance = None
        self.eps = 1e-8
        self.gamma = torch.nn.Parameter(torch.rand(size=(in_features,)))
        self.beta = torch.nn.Parameter(torch.zeros(size=(in_features,)))


    def forward(self, x):
        self.x = x
        self.meanvar = torch.mean(self.x, axis=0)
        self.variance = torch.var(self.x, axis=0)
        norm = ( self.x - self.meanvar ) / ( torch.sqrt(self.variance + self.eps) )
        self.output = self.gamma * norm + self.beta
        return self.output



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=24, out_features=32),
            BatchNorm(in_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
            BatchNorm(in_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=1)
        )


    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


class LossHuber(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, y_prim, y):
        loss = torch.mean(
            self.delta**2 * (torch.sqrt(1 + ((y - y_prim)/self.delta)**2) - 1.0)
        )
        return loss

model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = torch.nn.MSELoss()

loss_plot_train = []
loss_plot_test = []
r2_plot_train = []
r2_plot_test = []

for epoch in range(1, 110):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        r2s = []
        for x, y in dataloader:

            y_prim = model.forward(x)
            loss = loss_fn.forward(y_prim, y)

            np_loss = loss.data.numpy()
            losses.append(np_loss)

            np_y = y.data.numpy()
            np_y_prim = y_prim.data.numpy()

            r2 = 1 - (np.sum((np_y_prim - np_y)**2) / np.sum((np.mean(np_y) - np_y)**2))
            r2s.append(r2)

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            r2_plot_train.append(np.mean(r2s))
        else:
            loss_plot_test.append(np.mean(losses))
            r2_plot_test.append(np.mean(r2s))


    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]} R2_train: {r2_plot_train[-1]} R2_test: {r2_plot_test[-1]}')

    if epoch % 10 == 0:
        fig, axes = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout(pad=5)
        ax1 = axes[0]
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[1]
        ax1.plot(r2_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(r2_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("r2 score")
        plt.show()