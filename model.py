import torch
import torchvision
import torchvision.transforms as transforms
import numpy
import random
import torch.nn as nn
import torch.nn.functional as F
import time

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.Apool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5, stride = 2)
        self.fc1 = nn.Linear(720, 600)
        self.fc2 = nn.Linear(600, 150)
        self.fc3 = nn.Linear(150, 20)
        self.fc4 = nn.Linear(20,3)
        self.fc5 = nn.Linear(720, 600)
        self.fc6 = nn.Linear(600, 150)
        self.fc7 = nn.Linear(150, 20)
        self.fc8 = nn.Linear(20,4)

        self.res1 = nn.Sequential(nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1)
                                , nn.ELU(), 
                                nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1))
        self.res2 = nn.Sequential(nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1)
                                , nn.ELU(), 
                                nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1))
        self.res3 = nn.Sequential(nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1)
                                , nn.ELU(), 
                                nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1))
        self.res4 = nn.Sequential(nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1)
                                , nn.ELU(), 
                                nn.Conv2d(12,12,kernel_size = 3, stride = 1, padding = 1))
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = F.elu(self.res1(x) + x)
        x = F.elu(self.res2(x) + x)
        x = F.elu(self.res3(x) + x)
        x = F.elu(self.res4(x) + x)
        x = self.Apool(x)
        x1 = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.elu(self.fc1(x1))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        y = F.elu(self.fc5(x1))
        y = F.elu(self.fc6(y))
        y = F.elu(self.fc7(y))
        y = self.fc8(y)
        y = F.softmax(y, dim=-1)

        return x, y

net = Net()


if __name__ == "__main__":
    print(net.parameters)