import os
os.path
img_dir = "./data/img/"
cmd_dir = "./data/cmd/"
import cv2
from model import Net
import torch
import time
net = Net()
net = net.cuda()

if __name__ == "__main__":
    # file_names = os.listdir("./data")
    # for file_name in file_names:
    #     tmp = os.listdir("./data/" + file_name)
    #     print(file_name, len(tmp))
    #     if file_name == 'img':
    #         for x in tmp:
    #             a = x.split(".png")[0]
    #             print(a)
    # print(file_names)

    x = cv2.imread("./data/img/1/2.png")
    x = torch.tensor(x).cuda()
    print(x.shape)
    x = x.permute(2, 1, 0) / 255.0
    print(x.shape)
    t1 = time.time()
    net.eval()
    y=net(x)
    time.sleep(0.1)
    print(time.time() - t1)


