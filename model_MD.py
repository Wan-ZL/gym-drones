import numpy as np

class Mission_Drone:
    def __init__(self, ID):
        self.type = "MD"
        self.ID = ID
        self.type = 'MD'
        self.battery_max = 10.0
        self.battery = self.battery_max  # battery level
        self.died = False           # drone died when battery goes to zero
        self.charging = False
        self.charging_rate = 0.1
        self.consume_rate = 0.01
        self.low_battery_thres = 0.1
        self.signal_level = 10  # range [1,10]
        self.maximum_signal_radius = 2
        self.xyz = np.zeros(3)      # MD location (location of destination)
        self.xyz_temp = self.xyz.copy()    # intermediate location to destination

        self.compromised = False
        self.surveyed_time = 0
        self.surveyed_cell = np.array([])     # cell ID goes here

    def battery_update(self):       # consume energy or charging
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
                print(f"{self.type} {self.ID} is FULL")



    def no_mission(self):
        if np.array_equal(self.xyz[:2], np.zeros(2)):
            self.charging = True

    def go_charging(self):
        self.xyz[:2] = np.zeros(2)
        self.charging = True

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



        