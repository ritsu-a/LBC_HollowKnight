# camera-ready

import sys

from dataset import HKDataset
from model import Net
from utils.utils import add_weight_decay

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

import time

# NOTE! NOTE! change this to not overwrite all log data when you train the model:

img = cv2.imread("./data/img/1/32.png")
img = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
img = torch.tensor(img)
img = img.permute(2, 1, 0) / 255.0

with open("./data/cmd/1/32.log", "rb") as file:
    op, di = pickle.load(file)

print(op, di)
from dataset import userinput
print(userinput(op, di))
img = img.cuda()
img = img.unsqueeze(0)
print(img.shape)


model_id = "1"

num_epochs = 30
batch_size = 4
learning_rate = 0.001

train_dataset = HKDataset()
print(len(train_dataset))

net = Net()
net = net.cuda()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=1)

params = add_weight_decay(net, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

# loss function
criterion = nn.MSELoss()

epoch_losses_train = []
epoch_losses_val = []
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    net.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    xall = np.zeros((3))
    yall = np.zeros((4))
    with tqdm(train_loader, desc="training") as pbar:
        for imgs, cmds in pbar:
        #current_time = time.time()

            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            x, y = net(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            a, b = cmds
            x = x.cpu()
            y = y.cpu()
            xall += x.sum(dim=0).detach().numpy()
            yall += y.sum(dim=0).detach().numpy()
            # compute the loss:
            loss = criterion(x, a) + criterion(y, b)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)
            pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" %
                                     (epoch + 1, np.mean(batch_losses),
                                      optimizer.param_groups[0]['lr']))

            #print (time.time() - current_time)

    torch.save(net.state_dict(), f"/home/pyshi/tmp/dl/models/save{epoch}.pth")
    print(xall / 769)
    print(yall / 769)

    img = cv2.imread("./data/img/1/32.png")
    img = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img)
    img = img.permute(2, 1, 0) / 255.0

    with open("./data/cmd/1/32.log", "rb") as file:
        op, di = pickle.load(file)

    print(op, di)
    from dataset import userinput
    print(userinput(op, di))
    img = img.cuda()
    img = img.unsqueeze(0)
    print(img.shape)

    net.eval()
    x, y = net(img)
    x = x.cpu()
    y = y.cpu()
    print(x, y)
