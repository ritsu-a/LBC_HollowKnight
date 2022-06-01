from torch.utils.data import Dataset
import torch
import csv
import cv2
import numpy as np
import os
import pickle

def userinput(operations, directions):
    if 'C' in operations:
        a = torch.tensor([1.,0.,0.])
    elif 'X' in operations:
        a = torch.tensor([0.,1.,0.])
    else:
        a = torch.tensor([0.,0.,1.])
    if 'Left' in directions:
        b = torch.tensor([1.,0.,0.,0.])
    elif 'Right' in directions:
        b = torch.tensor([0.,1.,0.,0.])
    elif 'Up' in directions:
        b = torch.tensor([0.,0.,1.,0.])
    else:
        b = torch.tensor([0.,0.,0.,1.])
    return a, b

class HKDataset(Dataset):

    def __init__(self,
                 device="cpu"):

        self.base_img_dir = "./data/img/"
        self.base_cmd_dir = "./data/cmd/"

        self.pairs = []
        self.img_pths = []
        self.cmd_pths = []
        
        self.baseimg = cv2.imread("./data/img/1/32.png")
        self.baseimg = torch.tensor(self.baseimg)
        self.baseimg = self.baseimg.permute(2, 1, 0) / 255.0

        img_dirs = os.listdir(self.base_img_dir)
        for img_dir_id in img_dirs:
            img_dir = self.base_img_dir + img_dir_id + "/"
            cmd_dir = self.base_cmd_dir + img_dir_id + "/"
            img_pths = os.listdir(img_dir)

            for img_pth in img_pths:
                id = img_pth.split(".png")[0]
                imgpth = img_dir + img_pth
                self.img_pths.append(id + ".png")
                self.cmd_pths.append(id + ".log")

                with open(cmd_dir + id + ".log", "rb") as file:
                    operations, directions = pickle.load(file)
                
                #print(id, operations, directions)
                cmd = userinput(operations, directions)

                self.pairs.append((imgpth, cmd))

        self.device = device

    def __len__(self):
        return len(self.pairs)

    # @profile
    def __getitem__(self, index):
        imgpth, cmd = self.pairs[index]

        img = cv2.imread(imgpth)
        img = cv2.resize(img, (400, 200), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img)
        img = img.permute(2, 1, 0) / 255.0 

        return (img, cmd)

