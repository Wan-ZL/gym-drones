class attacker_model:
    def __init__(self):
        self.xyz_axis = [0,0,0]             # attacker's location
        self.maximum_signal_radius_HD = 3   # signal radius of HD
        self.sg_HD = 5                      # defense strategy, range [0, 10]


    def update_location(self, x_axis, y_axis, z_axis):
        self.xyz_axis = (x_axis, y_axis, z_axis)