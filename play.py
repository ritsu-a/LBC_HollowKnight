import numpy as np
import os
import cv2
import time
import collections
import matplotlib.pyplot as plt
import torch

import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import take_action, restart,take_direction, TackAction
from Tool.WindowsAPI import grab_screen
from Tool.GetHP import Hp_getter
from Tool.UserInput import User
from Tool.FrameBuffer import FrameBuffer

from model import Net

window_size = (0,0,1920,1017)
station_size = (230, 230, 1670, 930)

HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 7
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)


action_name = ["Attack", "Attack_Down", "Attack_Up",
           "Short_Jump", "Mid_Jump", "Skill", "Skill_Up", 
           "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1


def sample(operations, directions):
    print(operations, directions)
    move = 0
    action = 0
    l,r,u,d = directions
    c,x,z = operations
    if l > r:
        if l>u:
            if l>d:
                move = 0
            else:
                move = 3
        else:
            if u>d:
                move = 2
            else:
                move = 3
    else:
        if r>u:
            if r>d:
                move = 1
            else:
                move = 3
        else:
            if u>d:
                move = 2
            else:
                move = 3
    if c>x:
        if c>z:
            action = 0
        else:
            action = 2
    else:
        if x>z:
            action = 1
        else:
            action = 2 
    if action == 0:
        if z > 0.3:
            action = 3
        else:
            action = 2
    elif action == 1:
        if u>0.1:
            action = 1
        else:
            action = 0
    else:
        action = 6

    p = np.random.rand()
    if p < 0.2:
        action = 3
    elif p < 0.4:
        action = 6
    print(move, action)
    return move, action


def run_episode(net, paused):
    restart()
    # learn while load game
    time.sleep(2)
    
    step = 0
    done = 0
    total_reward = 0

    thread1 = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread1.start()

    
    while True:
        step += 1
        # in case of do not collect enough frames
        while(len(thread1.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.1)

        stations = thread1.get_buffer()
        
        print(len(stations))
        print(stations[0].shape)
        for i in range(4):
            cv2.imwrite(f"station{i}.png", stations[i])
            stations[i] = stations[i].transpose(2, 1, 0)


        x, y = net(torch.tensor(stations) / 255.0)
        move, action = sample(x.detach().numpy().mean(axis=0) - [0.06143111,0.30603395,0.63253494]
                            , y.detach().numpy().mean(axis=0) - [0.37749672,0.32761124 + 0.054,0.02640672,0.26848531])

        take_direction(move)
        take_action(action)
    thread1.stop()
    return 


if __name__ == '__main__':

    
    net = Net()
    net.load_state_dict(torch.load("save1.pth"))
    net.eval()

    print(net.parameters)


    paused = True
    paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    while episode < max_episode:    # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1     
        total_reward, total_step, PASS_COUNT = run_episode(net, paused)
                
        print("Episode: ", episode, ", pass_count: " , PASS_COUNT, ", average remind hp:", total_remind_hp / episode)

