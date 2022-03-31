import numpy as np

class Drone:
    def __init__(self, ID, update_freq):
        self.ID = ID
        self.type = 'Drone'
        self.update_freq = update_freq
        self.died = False  # drone died when battery goes to zero
        # self.charging = True
        self.xyz = np.zeros(3)  # Droen location (location of destination)
        self.xyz_temp = self.xyz.copy()  # intermediate location to destination
        self.speed_per_frame_max = 1000
        self.crashed = False             #
        self.in_GCS = True
        self.battery_max = 100.0         # battery level
        self.battery = self.battery_max
        self.consume_rate = 0.01
        # self.low_battery_thres = self.update_freq * self.consume_rate + 0.1  # this value is based on the consumption in one round
        self.max_signal = 20                # unit: dBm (fixed)
        self.signal = self.max_signal       # unit: dBm
        self.signal_radius = 1000           # unit meter
        self.been_attack_record = (0,0)     # the first element is # of success, the second element is # of failure.

    # def battery_update(self):       # consume energy or charging (True means drone is ready (recalculate trajectory))
    #     if not self.charging and not self.crashed:
    #         if self.battery > 0:
    #             self.battery -= self.consume_rate
    #         else:
    #             self.crashed = True             # if not charging, battery empty, then drone crash
    #     elif self.charging:
    #         if self.battery < self.battery_max:         # charging
    #             self.battery += self.charging_rate
    #         else:                                       # battery full
    #             if not np.array_equal(self.xyz[:2], np.zeros(2)):
    #                 self.charging = False
    #             return True
    #     return False
    def battery_update(self):       # consume energy or charging (True means drone is ready (recalculate trajectory))
        if self.crashed:        # ignore crashed drone
            return

        if self.in_GCS:
            if self.battery < self.battery_max: # drone in GCS with not full battery will charge
                self.battery += self.charging_rate
        else:                                   # any drone not in GCS consume energy
            self.battery -= self.consume_rate


    # check if a drone should change from normal condition to crashed condition
    def new_crash(self, height_z):
        if self.crashed:
            return False
        if not self.in_GCS:     # drone in GCS always safe
            if self.battery <= 0:
                print("\n====Mission Drone crashed by zero battery====, ID:", self.ID, self.type, "\n")
                self.crashed = True
                return True
            if height_z < 0.1:
                self.crashed = True
                print("\n====Mission Drone crashed====, ID:", MD.ID, "\n")  # TODO: add crash for HD
                self.crashed = True
                return True
        return False




        # if not self.in_GCS and not self.crashed:
        #     if self.battery > 0:
        #         self.battery -= self.consume_rate
        # elif self.charging:
        #     if self.battery < self.battery_max:         # charging
        #         self.battery += self.charging_rate
        #     else:                                       # battery full
        #         if not np.array_equal(self.xyz[:2], np.zeros(2)):
        #             self.charging = False
        #         return True
        # return False

    def assign_destination(self, xyz_destination):
        if self.battery < self.low_battery_thres:   # low battery go charging
            self.go_charging()
        elif not self.crashed:
            self.xyz = xyz_destination

    # assigning x y but keep z
    def assign_destination_xy(self, xy_destination):
        if self.battery < self.low_battery_thres:  # low battery go charging
            self.go_charging()
        elif not self.crashed:
            self.xyz[:2] = xy_destination

    # this value is based on the consumption in one round
    # @property make this function as variable so '()' is not required when calling it (same as Getter in Java)
    @property
    def low_battery_thres(self):
        return self.update_freq * self.consume_rate + 0.1

    def __repr__(self):
        return str(vars(self))


    def condition_check(self):
        if not self.crashed:
            if np.array_equal(self.xyz[:2], np.zeros(2)):
                self.in_GCS = True
            else:
                self.in_GCS = False

    def go_charging(self):
        self.xyz[:2] = np.zeros(2)
        # self.charging = True
