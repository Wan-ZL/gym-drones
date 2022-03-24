import numpy as np
from model_Drone import Drone

class Honey_Drone(Drone):
    def __init__(self, ID, update_freq):
        Drone.__init__(self, ID, update_freq)
        self.type = "HD"
        self.type = "HD"
        self.battery_max = 100.0
        self.battery = self.battery_max  # battery level
        self.charging_rate = 0.01
        self.consume_rate = 0.005
        self.maximum_signal_radius = 3
        self.protecting = np.array([])        # the MD (ID) protecting now
