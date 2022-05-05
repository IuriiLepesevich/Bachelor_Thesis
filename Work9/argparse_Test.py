import pandas as pd
import time
from copy import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse

import torchvision

from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams


from torchvision.datasets import FashionMNIST
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-batch_mode', default=f'Simple', type=str)
parser.add_argument('-run_name', default=f'run', type=str)
parser.add_argument('-sequence_name', default=f'reg3', type=str)
parser.add_argument('-epochs', default=15, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('--local_rank', default=0, type=int)

args, unknown_args = parser.parse_known_args()

plt.rcParams["figure.figsize"] = (12, 14) # size of window
plt.style.use('dark_background')

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_TEST_SPLIT = 0.8


def O_fun(x):
    return np.sqrt( np.sum( np.power( (x - np.mean(x, axis=0)), 2 ), axis=0 ) / len(x) )

def stand(x):
    return ( x - np.mean(x, axis=0) ) / (O_fun(x) + 1e-8)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        X = np.array(pd.read_csv("./star_classification.csv"))

        Cols_Alpha_Z = X[:, 2:8]
        Cols_Cam_col = X[:, 10:11]
        Cols_Red_shift_MJD = X[:, 14:17]

        temp = np.concatenate((Cols_Alpha_Z, Cols_Cam_col, Cols_Red_shift_MJD), axis=1)

        self.X = np.array(temp).astype(np.float32)
        #plt.hist(self.X[:, 4], bins=15)
        #plt.show()
        X_std = stand(self.X)
        self.X = torch.FloatTensor(X_std)

        self.Y = np.array(X[:, 1]).astype(np.float32)
        #plt.hist(self.Y, bins=20)
        #plt.show()
        Y_std = stand(self.Y)
        self.Y_mean = np.mean(self.Y)
        self.Y = torch.FloatTensor(Y_std)

        self.labels = np.expand_dims(np.unique(X[:, 13:14]), axis=0)
        self.X_classes = np.empty(len(X[:, 13:14]))
        for i in range(len(X[:, 13:14])):
            if X[i, 13:14] == 'GALAXY':
                self.X_classes[i] = 0
            elif X[i, 13:14] == 'QSO':
                self.X_classes[i] = 1
            else:
                self.X_classes[i] = 2

        self.X_classes = np.expand_dims(self.X_classes, axis=1)
        self.X_classes = self.X_classes.astype(int)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.X_classes[idx]), np.expand_dims(self.Y[idx], axis=-1),


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


class BatchNormLast(torch.nn.Module):
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
        if self.training:
            self.meanvar = torch.mean(self.x, axis=0)
            self.variance = torch.var(self.x, axis=0)
        norm = (self.x - self.meanvar) / (torch.sqrt(self.variance + self.eps))
        self.output = self.gamma * norm + self.beta
        return self.output



class BatchNormSimple(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.x = None

        self.mean_train = None
        self.mean_test = None

        self.var_train = None
        self.var_test = None

        self.mean_train_sum = np.array([])
        self.var_train_sum = np.array([])

        self.eps = 1e-8
        self.gamma = torch.nn.Parameter(torch.rand(size=(in_features,)))
        self.beta = torch.nn.Parameter(torch.zeros(size=(in_features,)))

        self.mean_train_sum = np.zeros((in_features,))


    def forward(self, x):
        self.x = x
        if self.training:
            self.mean_train = torch.mean(self.x, axis=0)
            self.var_train = torch.mean((self.x - self.mean_train) ** 2)

            self.mean_train_sum += self.mean_train.cpu().data.numpy()

            norm = (self.x - self.mean_train) / (torch.sqrt(self.var_train + self.eps))
            self.output = self.gamma * norm + self.beta
        else:
            tm = torch.FloatTensor(self.mean_train_sum)
            tv = torch.FloatTensor(self.mean_train_sum)

            self.mean_test = torch.mean(tm)
            self.var_test = torch.mean(tv)

            norm = (self.x - self.mean_test) / (torch.sqrt(self.var_test + self.eps))
            self.output = self.gamma * norm + self.beta
        return self.output



class BatchNormRunningMean(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.x = None

        self.mean_test = torch.rand(size=(in_features,))
        self.var_test = torch.rand(size=(in_features,))

        self.eps = 1e-8
        self.alpha = 0.9
        self.gamma = torch.nn.Parameter(torch.rand(size=(in_features,)))
        self.beta = torch.nn.Parameter(torch.zeros(size=(in_features,)))


    def forward(self, x):
        self.x = x
        if self.training:
            mean_train = torch.mean(self.x, axis=0)
            var_train = torch.var(self.x, axis=0)
            self.mean_test = self.alpha * self.mean_test + (1 - self.alpha) * mean_train
            self.var_test = self.alpha * self.var_test + (1 - self.alpha) * var_train

        norm = (self.x - self.mean_test) / (torch.sqrt(self.var_test + self.eps))
        self.output = self.gamma * norm + self.beta
        return self.output



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        Norm = BatchNormLast
        if args.batch_mode == 'Simple':
            Norm = BatchNormSimple
        elif args.batch_mode == 'RunningMean':
            Norm = BatchNormRunningMean

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=10 + 2 * len(dataset_full.labels), out_features=24),
            torch.nn.ReLU(),
            #Norm(in_features=24),
            torch.nn.Linear(in_features=24, out_features=16),
            torch.nn.ReLU(),
            #Norm(in_features=16),
            torch.nn.Linear(in_features=16, out_features=1)
        )

        self.embs = torch.nn.ModuleList()
        for labels in dataset_full.labels:
            self.embs.append(
                torch.nn.Embedding(
                    num_embeddings=len(labels),
                    embedding_dim=2
                )
            )


    def forward(self, x, x_classes):
        list_x_emb = []
        for i, emb in enumerate(self.embs):
            x_emb_each = emb.forward(x_classes[:, i])
            list_x_emb.append(x_emb_each)
        x_emb = torch.cat(list_x_emb, dim=-1)
        x_cat = torch.cat([x, x_emb], dim=-1)
        y_prim = self.layers.forward(x_cat)
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

class TensorBoardSummaryWriter(SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=10, filename_suffix='',
                 write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue,
                         flush_secs, filename_suffix, write_to_disk,
                         log_dir, **kwargs)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        hparam_dict = copy(hparam_dict)
        for key in list(hparam_dict.keys()):
            if type(hparam_dict[key]) not in [float, int, str]:
                del hparam_dict[key]
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)


summary_writer = TensorBoardSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}_{int(time.time())}'
)

for epoch in range(1, args.epochs + 1):
    metrics = {
    }
    print(f'\n{epoch-1}')
    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader == dataloader_train:
            mode = 'train'
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            mode = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)

        metrics[f'{mode}_loss'] = []
        metrics[f'{mode}_r2'] = []
        for x, x_classes, y in tqdm(dataloader, desc=mode):

            y_prim = model.forward(x, x_classes)
            loss = loss_fn.forward(y_prim, y)

            np_y = y.data.numpy()
            np_y_prim = y_prim.data.numpy()

            r2 = 1 - (np.sum((np_y_prim - np_y)**2) / (np.sum(dataset_full.Y_mean - np_y)**2))

            metrics[f'{mode}_loss'].append(loss.cpu().item())
            metrics[f'{mode}_r2'].append(r2)

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    metrics_mean = {}
    for key in metrics:
        mean_value = np.mean(metrics[key])
        metrics_mean[key] = mean_value

        print(f'{key}: {mean_value}')

        summary_writer.add_scalar(
            scalar_value=mean_value,
            tag=key,
            global_step=epoch
        )

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metrics_mean,
        name=args.run_name,
        global_step=epoch
    )
    summary_writer.flush()