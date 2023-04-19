#!.venv/bin/python

'''.
An MNIST Generator
This was created in a live coding session for the CdA Machine Learners Group
* NOTE: in it's current state, we flipped the inputs and outputs, so it generates mnist images from one-hot vectors.
* NOTE: you'll have to rewrite the dataloader code a bit since it relies on a local dependency. Consider using: https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
'''


import numpy as np
import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import einsum
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import torch, sys

SEED = 0
torch.random.manual_seed(SEED)
np.random.seed(SEED)


##################################################
# PARAMETERS

LR = 5e-3
BATCH_SIZE = 100
DEVICE = 'cuda' if len(sys.argv) <= 1 else sys.argv[1]


##################################################
# DATA

data_path = 'data'

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
            nn.Linear(IMG_SIZE*IMG_SIZE, 128),
            # nn.Tanh(),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
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

epochs = 5
for t in range(epochs):
    print(f'Epoch {t+1}\n------------------------------')
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print('Trained.')


# for i in range(10):
#     n = onehot(i, 10).to('cuda')
#     img = model(n.unsqueeze(0)).reshape((28, 28))
#     plt.subplot(4, 4, i+1)
#     plt.imshow(img.detach().cpu().numpy())
# plt.show()


##################################################
##################################################

import numpy as np
import cv2, math, random


def image_to_array(img, size=28):
    # Convert image to array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.bitwise_not(img)
    #img = img / 255.0
    #img = img.reshape(1, size, size, 1)

    ret = []
    for i in range(size):
        for j in range(size):
            ret.append(math.ceil(img[i][j]))

    #cv2.imshow("Before NN", img)

    return ret

#create a 512x512 black image
nn_img = np.zeros((1024,256,3), np.uint8)
draw = np.zeros((1024,1024,3), np.uint8)

def drawCircles(img, ary):
    #img = img.copy()
    ary = ary.tolist()
    # Calc some dimensions
    init = 30
    height = img.shape[0]
    step = (height - init * 2) / len(ary)

    #draw a circle
    x = math.floor(img.shape[1] / 2)
    y = step / 2 + init / 2
    for t in ary:
        color = (0, math.floor(t * 255), math.floor((1 - t) * 255))

        #non filled circle
        cv2.circle(img, (x, math.floor(y)), math.floor(step * 0.3), (255,0,0), 3)
        #filled circle
        cv2.circle(img, (x, math.floor(y)), math.floor(step * 0.3), color, -1)

        y += step

    return img

def draw_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
        (w,h,d) = draw.shape
        cv2.rectangle(draw, (0,0), (w,h), (0,0,0), -1)

        ary = torch.zeros(10)# [random.random() for i in range(10)]
        drawCircles( nn_img, ary)

    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(draw, (x, y), brush_size, (255, 255, 255), -1)

        # Run the neural network
        # print( image_to_array(draw) )
        x = image_to_array(draw)
        # print(x)
        x = torch.tensor(x).to(DEVICE).unsqueeze(0) / 255
        x += torch.randn_like(x) / 1000
        # ary = [random.random() for i in range(10)]
        ary = model(x)[0]
        drawCircles( nn_img, ary)

# First draw
model.eval()
ary = torch.zeros(10)# [random.random() for i in range(10)]
drawCircles( nn_img, ary)

brush_size = 25

cv2.namedWindow(winname="Draw a number")
cv2.setMouseCallback("Draw a number", draw_mouse)

while True:
    cv2.imshow("Draw a number", draw)
    cv2.imshow("AI Output", nn_img)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
