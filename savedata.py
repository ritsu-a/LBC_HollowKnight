# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import cv2
import time
import collections
import matplotlib.pyplot as plt


import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import Move_Left, take_action, restart,take_direction, TackAction
from Tool.WindowsAPI import grab_screen
from Tool.WindowsAPI import key_check
from Tool.GetHP import Hp_getter
from Tool.UserInput import User
from Tool.FrameBuffer import FrameBuffer
import pickle

window_size = (0,0,1920,1017)
station_size = (230, 230, 1670, 930)

HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 7
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)

MEMORY_SIZE = 200  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 24  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 10  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.00001  # 学习率
GAMMA = 0

action_name = ["Attack", "Attack_Up",
           "Short_Jump", "Mid_Jump", "Skill_Up", 
           "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1



i = 3 #this define save data in which folder

def run_episode(paused):
    restart()
    step = 0
    while step <= 1000:
        if step < 5:
            time.sleep(1)
            step += 1
            continue
        print(step)
        step += 1
        station = grab_screen()
        #cv2.imwrite(f"./data/img/{i}/{step-5}.png", station)
        operations, direction = key_check()
        print(operations, direction)
        with open(f"./data/cmd/{i}/{step-5}.log", "wb") as file:
            pickle.dump((operations, direction), file)
        time.sleep(0.1)
    return


if __name__ == '__main__':

    # In case of out of memory
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True      #程序按需申请内存
    sess = tf.compat.v1.Session(config = config)

    # paused at the begining
    paused = True
    paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0                                       # pass count
    while episode < max_episode:    # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1     
        print(episode)
        # if episode % 20 == 1:
        #     algorithm.replace_target()

        run_episode(paused)
        
        
        print("Episode: ", episode)

