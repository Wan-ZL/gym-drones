class Honey_Drone:
    def __init__(self, ID):
        self.type = "HD"
        self.ID = ID
        self.type = "HD"
        self.battery = 100.0            # battery level
        self.charging = False
        self.charging_rate = 0.1
        self.consume_rate = 0.005
        self.maximum_signal_radius = 3
        self.xyz = [0, 0, 0]       # HD location (location of destination)

        self.signal_level = 10      # range [1,10]
        self.protecting = []        # the MD (ID) protecting now

    def battery_update(self):       # consume energy or charging
        if not self.charging:
            self.battery -= self.consume_rate
        else:
            self.battery += self.charging_rate
