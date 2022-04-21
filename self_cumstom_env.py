from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model

class DroneSim(Env):
    def __init__(self, defender):
        self.defender = defender
        self.action_space = Discrete(self.defender.strategy)
        self.mission_time_max = self.defender.system.mission_max_duration
        self.mission_time = 0       # state - 1
        self.surv_complete = 0      # state - 2
        # [time duration, ratio of mission complete]
        self.observation_space = Box(low=np.array([0., 0.]), high=np.array([self.defender.system.mission_max_duration, 1.]))

    def step(self, action):
        self.defender.set_strategy(action)
        self.defender.action()

        self.mission_time += 1

        state = [self.mission_time, self.surv_complete]
        reward = 0
        if self.mission_time < 2: #self.mission_time_max:
            done = False
        else:
            done = True
        info = {}
        return state, reward, done, info


    def render(self, *args):
        pass

    def reset(self, *args):
        self.mission_time = 0
        self.surv_complete = 0
        return [self.mission_time, self.surv_complete]

class DQN(nn.Module):
    def __init__(self, obser_space, action_space,
                 batch_size=32,
                 learning_rate=0.01,
                 epsilon=0.9,
                 gamma=0.9,
                 target_replace_iter=100,
                 memory_size=2000):
        super(DQN, self).__init__()
        self.eval_net = self.build_Net(obser_space, action_space)
        self.target_net = self.build_Net(obser_space, action_space)

        self.dim_state = obser_space  # 状态维度
        self.n_actions = action_space  # 可选动作数
        self.batch_size = batch_size  # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # probability of NOT randomly selection action. 贪婪系数
        self.gamma = gamma  # 回报衰减率
        self.memory_size = memory_size  # 记忆库的规格
        self.taget_replace_iter = target_replace_iter  # update target_net every this number (of time). target网络延迟更新的间隔步数
        self.learn_step_counter = 0  # count the number of time passed. 在计算隔n步跟新的的时候用到
        self.memory_counter = 0  # 用来计算存储索引
        # each horizontal line contain: [previous state, parameter of pre state, parameter of new state, new state]
        self.memory = np.zeros((self.memory_size, self.dim_state * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # optimizer
        self.loss_func = nn.MSELoss()       # loss function


    def build_Net(self, obser_space, action_space):
        net = nn.Sequential(
            nn.Linear(obser_space, 16),
            nn.ReLU(),
            nn.Linear(16, action_space)
        )
        return net

    # def forward(self, x):
    #     return self.net(x)

    def choose_action(self, x):
        X = torch.unsqueeze(torch.FloatTensor(x), 0)    # transfer state to tensor
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(X)              # forward in network
            action = torch.max(action_value, 1)[1]          # get action with max value
            action = int(action)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # back-propagattion
    def learn(self):
        # update target_net parameter every 'target_replace_iter' time
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # get data from memory
        data_size = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter

        sample_index = np.random.choice(data_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        # [0 : b_s : b_a : b_r: b_s_]
        b_s = torch.FloatTensor(b_memory[:, :self.dim_state])       # previous state
        b_a = torch.LongTensor(b_memory[:, self.dim_state:self.dim_state + 1].astype(int))  # integer used for torch.gather dimension
        b_r = torch.FloatTensor(b_memory[:, self.dim_state + 1:self.dim_state + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.dim_state:])     # next state

        # calculate LOSS
        q_eval = self.eval_net(b_s).gather(1, b_a)  # get result for previous state
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        # back propagation
        self.optimizer.zero_grad()  # since gradient value is accumulated, reset it before use.
        loss.backward()
        self.optimizer.step()

    # save one state to memory
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DQN().to(device)
    # print(model)
    system = system_model()
    defender = defender_model(system)
    env = DroneSim(defender)
    print("action space", env.action_space.sample())
    print("observation space", env.observation_space.sample())

    # test environment and DQN
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    print(model)

    for episode in range(10):
        state_pre = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            a = model.choose_action(state_pre)      # select an action
            state_new, reward, done, info = env.step(a)     # execute action
            model.store_transition(state_pre, a, reward, state_new)     # save informatin to memory
            score += reward     # accmulate reward

            # train network when memory is full
            if model.memory_counter > model.memory_size:
                model.learn()

            state_pre = state_new

        print('Episode:{} Score:{}'.format(episode, score))


    # testing neural network
    print("Testing . . .")
    score_list = []
    for episode in range(5):
        state_pre = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            a = model.choose_action(state_pre)
            state_new, reward, done, info = env.step(a)  # execute action
            score += reward

        score_list.append(score)
        print('Test: Episode:{} Score:{}'.format(episode, score))

    env.close()
    print(np.average(score_list))













