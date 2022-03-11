class Honey_Drone:
    def __init__(self, ID):
        self.ID = ID
        self.type = "HD"
        self.battery = 1.0      # battery level
        self.protecting = []    # the MD (ID) protecting now
