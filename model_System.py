import numpy as np
from model_MD import Mission_Drone
from model_HD import Honey_Drone

class system_model:
    def __init__(self):
        self.mission_complete = False
        self.target_map_size = 4
        self.map_ori_x = 1   # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.target_map_size,self.target_map_size))
        self.min_scan_requirement = 1
        self.num_MD = 3  # number of MD (in index, MD first then HD)
        self.num_HD = 1  # number of HD
        self.MD_dict = {}        # key is id, value is class detail
        self.HD_dict = {}
        self.assign_MD()
        self.assign_HD()
        self.MD_set = self.MD_dict.values()
        self.HD_set = self.HD_dict.values()

    def not_scanned_map(self):
        return self.scan_map >= self.min_scan_requirement

    def print_MDs(self):
        for MD in self.MD_set:
            print(vars(MD))

    def print_HDs(self):
        for HD in self.HD_set:
            print(vars(HD))

    def location_backup_MD(self):


        pass

    def battery_consume(self):
        for MD in self.MD_set:
            MD.battery_update()
        for HD in self.HD_set:
            HD.battery_update()

    def assign_MD(self):
        for index in range(self.num_MD):
            self.MD_dict[index] = Mission_Drone(index)

    def assign_HD(self):
        for index in range(self.num_HD):
            self.HD_dict[index] = Honey_Drone(index)

    def sample_MD(self):
        return Mission_Drone(-1)

    def sample_HD(self):
        return Honey_Drone(-1)

    def check_mission_complete(self):
        if np.all(self.scan_map > 1):
            self.mission_complete = True
            print("Mission Complete!! \n scanned map:")
            print(self.scan_map)
        return self.mission_complete

    # this is a fast way to check for saving running time
    def is_mission_complete(self):
        return self.mission_complete

    def update_scan(self, x_axis, y_axis, amount):
        x_axis = x_axis - self.map_ori_x
        y_axis = y_axis - self.map_ori_y
        if 0 <= x_axis and x_axis < self.target_map_size and 0 <= y_axis and y_axis < self.target_map_size:
            self.scan_map[x_axis, y_axis] += amount


