import numpy as np

class system_model:
    def __init__(self):
        self.mission_complete = False
        self.target_map_size = 4
        self.map_ori_x = 1   # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.target_map_size,self.target_map_size))

    def check_mission_complete(self):
        if min(self.scan_map) > 1:
            self.mission_complete = True

    def is_mission_complete(self):
        return self.mission_complete

    def update_scan(self, x_axis, y_axis, amount):
        if self.map_ori_x <= x_axis and x_axis < self.target_map_size and self.map_ori_y <= y_axis and y_axis < self.target_map_size:
            self.scan_map[x_axis, y_axis] += amount

