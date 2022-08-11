import numpy as np

class Drone:
    def __init__(self, ID, update_freq):
        self.print_debug = False
        self.ID = ID
        self.type = 'Drone'
        self.update_freq = update_freq
        self.died = False  # drone died when battery goes to zero
        # self.charging = True
        self.xyz = np.zeros(3)  # Droen location (location of destination)
        self.xyz_temp = self.xyz.copy()  # intermediate location to destination
        # TODO: drone crash when move long distance. Find a good way to control drone
        # TODO: drone not arrive before go next destination, solve this by keeping the same destination if target is not scan complete.
        self.speed_per_frame_max = 0.08     # this value obtained from experiment that drone doesn't crash for a 150 meter fly in one round
        self.crashed = False             #
        self.in_GCS = True
        self.battery_max = 100000.0         # battery level
        self.battery = self.battery_max
        self.E_P = 0.001
        self.E_C = 0.001
        self.E_R = 0.001
        self.consume_rate = self.E_P + self.E_C + self.E_R
        self.accumulated_consumption = 0        # this will show the total energy consumption in one episode
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
            return False

        if self.in_GCS:
            if self.battery < self.battery_max: # drone in GCS with not full battery will charge
                self.battery += self.charging_rate
                # send signal if battery full
                if self.battery >= self.battery_max:
                    return True
        else:                                   # any drone not in GCS consume energy
            self.battery -= self.consume_rate
            self.accumulated_consumption += self.consume_rate
        return False

    def consume_rate_update(self, DS_j):
        self.consume_rate = self.E_P + self.E_C + self.E_R * (DS_j/10)



    # check if a drone should change from normal condition to crashed condition
    def new_crash(self, xyz_current):
        if self.crashed:
            return False
        if not self.in_GCS:     # drone in GCS always safe
            if self.battery <= 0:
                if self.print_debug: print("\n====Drone crashed by zero battery====, ID:", self.ID, self.type, "\n")
                self.crashed = True
                return True
            if xyz_current[2] < 0.1:
                self.crashed = True
                if self.print_debug: print("\n====Drone crashed by zero height====, ID:", self.ID, self.type, xyz_current, "\n")
                if self.print_debug: print(self)
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
        return self.update_freq * (self.E_P + self.E_C + self.E_R) + 0.1

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
