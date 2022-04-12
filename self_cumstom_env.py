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

class HoneyDrone(Env):
    def __init__(self, defender):
        self.defender = defender
        self.action_space = Discrete(self.defender.strategy)
        self.mission_time = self.defender.system.mission_max_duration
        # [time duration, ratio of mission complete]
        self.observation_space = Box(low=np.array([0., 0.]), high=np.array([self.defender.system.mission_max_duration, 1.]))

    def step(self, action):
        self.defender.select_strategy(action)

    def render(self, *args):
        pass

    def reset(self, *args):
        self.mission_time = self.defender.system.mission_max_duration
        return

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    print(model)








