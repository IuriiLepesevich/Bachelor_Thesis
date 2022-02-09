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


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '.\For_modeling_temp.csv'
        with open(f'{path_dataset}', 'rb') as fp:
            X = pd.read_csv(fp)

        X = np.array(X)

        self.X = np.array(X[:, 2:]).astype(np.float32)
        self.X = F.normalize(torch.tensor(self.X))

        self.Y = np.array(X[:, 1]).astype(np.float32)
        self.Y = F.normalize(torch.tensor(self.Y), dim=0)

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


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=24, out_features=16),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=16, out_features=8),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=8, out_features=1)
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
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = LossHuber(delta=0.5)

loss_plot_train = []
loss_plot_test = []
r2_plot_train = []
r2_plot_test = []

for epoch in range(1, 1000):

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