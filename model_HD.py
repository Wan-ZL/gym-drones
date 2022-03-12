class Honey_Drone:
    def __init__(self, ID):
        self.ID = ID
        self.type = "HD"
        self.battery = 1.0      # battery level
        self.maximum_signal_radius = 3

        self.signal_level = 10  # range [1,10]
        self.protecting = []    # the MD (ID) protecting now
