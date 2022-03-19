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
        self.battery_max = 100.0         # battery level
        self.battery = self.battery_max
        self.consume_rate = 0.01
        # self.low_battery_thres = self.update_freq * self.consume_rate + 0.1  # this value is based on the consumption in one round

    # this value is based on the consumption in one round
    # @property make this function as variable so '()' is not required when calling it (same as Getter in Java)
    @property
    def low_battery_thres(self):
        return self.update_freq * self.consume_rate + 0.1

    def __repr__(self):
        return str(vars(self))

    def no_mission(self):
        if np.array_equal(self.xyz[:2], np.zeros(2)):
            self.charging = True

    def go_charging(self):
        self.xyz[:2] = np.zeros(2)
        self.charging = True
