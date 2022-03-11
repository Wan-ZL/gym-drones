class Mission_Drone:
    def __init__(self, ID):
        self.ID = ID
        self.type = 'MD'
        self.battery = 1.0      # battery level
        self.signal_level = 10  # range [1,10]
        self.maximum_signal_radius = 2

        self.compromised = False
        self.surveyed_time = 0
        self.surveyed_cell = []     # cell ID goes here
        