class Mission_Drone:
    def __init__(self, ID):
        self.ID = ID
        self.type = 'MD'
        self.battery = 1.0      # battery level
        self.compromised = False
        self.surveyed_time = 0
        self.surveyed_cell = []     # cell ID goes here
        