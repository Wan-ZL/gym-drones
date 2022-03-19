import numpy as np
from model_Drone import Drone

class Mission_Drone(Drone):
    def __init__(self, ID, update_freq):
        Drone.__init__(self, ID, update_freq)
        self.type = "MD"
        self.type = 'MD'
        self.battery_max = 30.0
        self.battery = self.battery_max  # battery level
        self.charging_rate = 0.02
        self.consume_rate = 0.01
        self.signal_level = 10  # range [1,10]
        self.maximum_signal_radius = 2
        self.surveyed_time = 0
        self.surveyed_cell = np.array([])     # cell ID goes here

    def battery_update(self):       # consume energy or charging (True means drone is ready (recalculate trajectory))
        if not self.charging and not self.compromised:
            if self.battery > 0:
                self.battery -= self.consume_rate
            else:
                self.died = True
        elif self.charging:
            if self.battery < self.battery_max:         # charging
                self.battery += self.charging_rate
            else:                                       # battery full
                self.charging = False
                print(f"{self.type} {self.ID} battery is FULL")
                return True
        return False

    def assign_destination(self, xyz_destination):
        if self.battery < self.low_battery_thres:   # low battery go charging
            self.go_charging()
        elif not self.charging or self.compromised:
            self.xyz = xyz_destination

    # assigning x y but keep z
    def assign_destination_xy(self, xy_destination):
        if self.battery < self.low_battery_thres:   # low battery go charging
            self.go_charging()
        elif not self.charging or self.compromised:
            self.xyz[:2] = xy_destination



        