# MNIST Classification Using PCA and VQC

## Import All Package

from os import listdir, path, system
import os
import time

import numpy as np

import torch
from torch import nn
from torch.nn import Linear
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pennylane as qml
import pickle

import argparse

## Load and Set Hyperparameter

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--bs", type=int, default=100)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--samples", type=int, default=6000)
parser.add_argument("--layers", type=int, default=5)
parser.add_argument("--numclass", type=int, default=10)
parser.add_argument(
    "--method",
    type=str,
    default='BasicEntanglerLayers',
    choices=['StronglyEntanglingLayers', 'BasicEntanglerLayers', 'layer'])
args = parser.parse_args()

print(args)

epochs = args.epochs
lr = args.lr
batch_size = args.bs
n_samples = args.samples

num_qubits = args.numclass
n_layers = args.layers
num_classes = args.numclass
use_GPU = False

batches = n_samples // batch_size

def time_now():
    return time.strftime(
        "%Y%m%d_%H%M%S",
        time.localtime()) + f'_{np.random.randint(0,1000):03d}'


## Set File Path

path = f'output_data/{args.method}_{num_qubits}_{n_layers}_{batch_size}_{lr}_{n_samples}_{time_now()}'
print(path)
os.makedirs(path, exist_ok=True)
path = "./" + path + "/"

## load MNIST Dataset

train_dataset = datasets.MNIST(root='./train_data', train=True, download=True)
test_dataset = datasets.MNIST(root='./test_data', train=False, download=True)

train_choose = np.random.choice(len(train_dataset.data), n_samples, False)
test_choose = np.random.choice(len(test_dataset.data), n_samples // 6, False)

X_train = train_dataset.data.view(-1, 784)
X_test = test_dataset.data.view(-1, 784)
y_train = train_dataset.targets
y_test = test_dataset.targets

## Apply PCA on MNIST Dataset

MNIST_PCA = PCA(n_components=num_qubits)
MNIST_PCA.fit(X_train)
X_train_pca = MNIST_PCA.transform(X_train)
X_test_pca = MNIST_PCA.transform(X_test)

## Rescaler to 0.1*pi ~ 1.9*pi

scaler = MinMaxScaler(feature_range=(0.05 * 2 * np.pi, 0.95 * 2 * np.pi))
scaler.fit(X_train_pca)
X_train_pca = scaler.transform(X_train_pca)[train_choose]
X_test_pca = scaler.transform(X_test_pca)[test_choose]

y_train = train_dataset.targets[train_choose]
y_test = test_dataset.targets[test_choose]

X_train_pca = torch.tensor(X_train_pca, requires_grad=False).float()
X_test_pca = torch.tensor(X_test_pca, requires_grad=False).float()

print(X_train_pca.shape, X_test_pca.shape)

## Define Dataloader

train_loader = torch.utils.data.DataLoader(list(zip(X_train_pca, y_train)),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(list(zip(X_test_pca, y_test)),
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0)

## Define VQC Model

dev = qml.device("default.qubit", wires=num_qubits)


def layer(weights):
    def _layer(W):
        for i in range(num_qubits):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if num_qubits > 2:
            qml.CNOT(wires=[num_qubits - 1, 0])

    for W in weights:
        _layer(W)


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    if args.method == 'layer':
        layer(weights)
    elif args.method == 'StronglyEntanglingLayers':
        qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    elif args.method == 'BasicEntanglerLayers':
        qml.BasicEntanglerLayers(weights, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_classes)]

if args.method == 'BasicEntanglerLayers':
    weight_shapes = {"weights": (n_layers, num_qubits)}
else:
    weight_shapes = {"weights": (n_layers, num_qubits, 3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

model = nn.Sequential(qlayer)
if torch.cuda.is_available() and use_GPU:
    model = model.cuda()

## loss Function and Optimizer

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fun = nn.CrossEntropyLoss()

## Train VQC Model

print("start")
ST = time.time()
train_loss, train_acc, test_loss, test_acc = [], [], [], []
for epoch in range(epochs):
    ### train set
    loss_sum, total_cnt, correct_cnt = 0, 0, 0
    model.train()
    for X, y in train_loader:
        X, y = Variable(X), Variable(y)
        if torch.cuda.is_available() and use_GPU:
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fun(out, y)
        loss.backward()
        optimizer.step()
        pred_y = torch.argmax(out, axis=1)
        total_cnt += y.data.shape[0]
        correct_cnt += (pred_y == y).sum()
        loss_sum += loss.item()
    train_loss.append(loss_sum / total_cnt)
    train_acc.append(float(correct_cnt) / total_cnt)
    print(
        f'Epoch {epoch + 1:3d}/{epochs:3d}\nTrain Loss: {train_loss[-1]:.10f} '
        f'Train Acc: {train_acc[-1]:.5f} Use Time: {time.time()-ST:5f}')
    ### test set
    loss_sum, total_cnt, correct_cnt = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = Variable(X), Variable(y)
            if torch.cuda.is_available() and use_GPU:
                X, y = X.cuda(), y.cuda()
            out = model(X)
            loss = loss_fun(out, y)
            pred_y = torch.argmax(out, axis=1)
            total_cnt += y.data.shape[0]
            correct_cnt += (pred_y == y).sum()
            loss_sum += loss.item()
        test_loss.append(loss_sum / total_cnt)
        test_acc.append(float(correct_cnt) / total_cnt)
    print(f'Test Loss: {test_loss[-1]:.10f} '
          f'Test Acc: {test_acc[-1]:.5f} Use Time: {time.time()-ST:5f}')

    with open(path + "_loss_acc.pkl", "wb") as fp:
        pickle.dump(
            {
                'Hyperparameters': args,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, fp)

    ## Plot Learning Curve
    
    plt.figure(dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(range(epoch + 1), test_loss, label='test')
    plt.plot(range(epoch + 1), train_loss, label='train')
    plt.legend()
    plt.title('loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch + 1), test_acc, label='test')
    plt.plot(range(epoch + 1), train_acc, label='train')
    plt.legend()
    plt.title('acc')
    plt.tight_layout()
    plt.savefig(path + 'loss_acc.png')
    torch.save(model.state_dict(), path + f"model_{epoch + 1}.pkl")
