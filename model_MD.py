class Mission_Drone:
    def __init__(self, ID):
        self.type = "MD"
        self.ID = ID
        self.type = 'MD'
        self.battery = 100.0      # battery level
        self.charging = False
        self.charging_rate = 0.1
        self.consume_rate = 0.01
        self.signal_level = 10  # range [1,10]
        self.maximum_signal_radius = 2
        self.xyz = [0, 0, 0]       # MD location (location of destination)
        self.xyz_temp = self.xyz    # intermediate location to destination

        self.compromised = False
        self.surveyed_time = 0
        self.surveyed_cell = []     # cell ID goes here

    def battery_update(self):       # consume energy or charging
        if not self.charging and not self.compromised:
            self.battery -= self.consume_rate
        elif self.charging:
            self.battery += self.charging_rate


        