import numpy as np

class player_model:
    def __init__(self, system):
        self.system = system  # save the pointer to system class
        self.xyz = [0, 0, 0]  # player's location
        self.strategy = 10
        self.min_strategy = 1  # minimum signal level player can choose
        self.max_strategy = 10  # maximum signal level player can choose

    def update_location(self, x_axis, y_axis, z_axis):
        self.xyz = (x_axis, y_axis, z_axis)