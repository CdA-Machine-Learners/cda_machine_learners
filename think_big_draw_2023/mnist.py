'''.
An MNIST Generator
This was created in a live coding session for the CdA Machine Learners Group
* NOTE: in it's current state, we flipped the inputs and outputs, so it generates mnist images from one-hot vectors.
* NOTE: you'll have to rewrite the dataloader code a bit since it relies on a local dependency. Consider using: https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
'''


import numpy as np
import torch as torch
import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import einsum
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

SEED = 0
torch.random.manual_seed(SEED)
np.random.seed(SEED)


##################################################
# PARAMETERS

LR = 5e-3
BATCH_SIZE = 100
DEVICE = 'cuda'


##################################################
# DATA

data_path = '/home/josh/_/beta/experiments/data'

def onehot(i, size):
    out = torch.zeros(size)
    out[i] = 1
    return out

try:
    test_dl
except:
    transform = torchvision.transforms.ToTensor()
    train_ds_pre = MNIST(data_path, train=True, transform=transform, download=True)
    train_ds = []
    for x, i in train_ds_pre:
        train_ds.append((x.flatten(), onehot(i, 10)))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds_pre = MNIST(data_path, train=False, transform=transform, download=True)
    test_ds = []
    for x, i in test_ds_pre:
        test_ds.append((x.flatten(), onehot(i, 10)))
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)


##################################################
# ARCHITECTURE

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        IMG_SIZE = 28
        self.w = nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.w(x)


##################################################
# TRAINING

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward Pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


##################################################
# BOMBS AWAY

model = Model().to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

epochs = 100
for t in range(epochs):
    print(f'Epoch {t+1}\n------------------------------')
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print('Done.')



# for i in range(10):
#     n = onehot(i, 10).to('cuda')
#     img = model(n.unsqueeze(0)).reshape((28, 28))
#     plt.subplot(4, 4, i+1)
#     plt.imshow(img.detach().cpu().numpy())
# plt.show()
