import numpy as np

class Drone:
    def __init__(self, ID, update_freq):
        self.ID = ID
        self.type = 'Drone'
        self.update_freq = update_freq
        self.died = False  # drone died when battery goes to zero
        self.charging = False
        self.xyz = np.zeros(3)  # Droen location (location of destination)
        self.xyz_temp = self.xyz.copy()  # intermediate location to destination
        self.compromised = False
        self.battery_max = 30.0         # battery level
        self.battery = self.battery_max

    def __repr__(self):
        return str(vars(self))


    def no_mission(self):
        if np.array_equal(self.xyz[:2], np.zeros(2)):
            self.charging = True

    def go_charging(self):
        self.xyz[:2] = np.zeros(2)
        self.charging = True
