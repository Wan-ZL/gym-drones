import numpy as np
from model_MD import Mission_Drone
from model_HD import Honey_Drone

class system_model:
    def __init__(self):
        self.update_freq = 500  # environment frame per round
        self.mission_Not_end = 2   # 2 means True, use 2 here to allow drone back to BaseStation in debug mode
        self.mission_success = False
        self.mission_max_status = self.mission_Not_end
        self.target_map_size = 6
        self.map_ori_x = 1   # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.target_map_size,self.target_map_size))
        self.min_scan_requirement = 1
        self.recalc_trajectory = False   # True: need recalculate trajectory
        self.num_MD = 3  # number of MD (in index, MD first then HD)
        self.num_HD = 1  # number of HD
        self.MD_dict = {}        # key is id, value is class detail
        self.HD_dict = {}
        self.MD_crashed_IDs = []   # includes MD compromised or crashed
        self.assign_MD()
        self.assign_HD()
        self.MD_set = self.MD_dict.values()
        self.HD_set = self.HD_dict.values()

    # scan count, and check if compromised
    def MD_environment_interaction(self, obs, scan_map):
        for MD in self.MD_set:
            if MD.compromised:
                continue
        # for i in range(num_MD):
            # print("obs[str(i)][state]", obs[str(i)]["state"][0:3].round(1))
            cell_x, cell_y, height_z = obs[str(MD.ID)]["state"][0:3]

            if height_z < 0.1:
                MD.compromised = True
                self.recalc_trajectory = True       # when a MD offline, recalculate trajectory
                print("Drone crashed, ID:", MD.ID)
                print("cell_x, cell_y, height_z", cell_x, cell_y, height_z)

            map_x_index = int(cell_x + 0.5)
            map_y_index = int(cell_y + 0.5)
            map_size_with_station = self.target_map_size + 1

            if 0 <= map_x_index and map_x_index < map_size_with_station and 0 <= map_y_index and map_y_index < map_size_with_station:
                scan_map[map_x_index, map_y_index] += 0.01
                self.update_scan(map_x_index, map_y_index, 0.01)
            else:
                print("ID, cell_x, cell_y, height_z", MD.ID, cell_x, cell_y, height_z)
                print("map_x_index_const, map_y_index_const", MD.ID, map_x_index, map_y_index)


    def not_scanned_map(self):
        return self.scan_map < self.min_scan_requirement

    def print_MDs(self):
        for MD in self.MD_set:
            print(vars(MD))

    def print_HDs(self):
        for HD in self.HD_set:
            print(vars(HD))

    def print_drones_battery(self):
        for MD in self.MD_set:
            print(f"MD {MD.ID} battery: {MD.battery}")
        for HD in self.HD_set:
            print(f"HD {HD.ID} battery: {HD.battery}")

    def location_backup_MD(self):


        pass

    def battery_consume(self):
        for MD in self.MD_set:
            if MD.compromised:
                continue
            if MD.battery_update():     # True means battery charging complete
                self.recalc_trajectory = True
        for HD in self.HD_set:          # True means battery charging complete
            if HD.battery_update():
                self.recalc_trajectory = True

    def assign_MD(self):
        for index in range(self.num_MD):
            self.MD_dict[index] = Mission_Drone(index, self.update_freq)

    def assign_HD(self):
        for index in range(self.num_HD):
            self.HD_dict[index] = Honey_Drone(index, self.update_freq)

    def sample_MD(self):
        return Mission_Drone(-1, self.update_freq)

    def sample_HD(self):
        return Honey_Drone(-1, self.update_freq)

    def check_mission_complete(self):
        # check if mission can end
        if np.all(self.scan_map > 1):
            self.mission_Not_end -= 1
            self.mission_success = True
        else:
            self.print_drones_battery()

        # check if mission end successfully
        if not self.mission_Not_end:
            if self.mission_success:
                print("Mission Complete!!")
                print("scanned map:\n", self.scan_map)
                self.print_MDs()
                self.print_HDs()
            else:
                print("Mission Fail :(")
                print("scanned map:\n", self.scan_map)
                self.print_MDs()
                self.print_HDs()

        return self.mission_Not_end

    # this is a fast way to check for saving running time
    def is_mission_Not_end(self):
        return self.mission_Not_end

    def update_scan(self, x_axis, y_axis, amount):
        x_axis = x_axis - self.map_ori_x
        y_axis = y_axis - self.map_ori_y
        if 0 <= x_axis and x_axis < self.target_map_size and 0 <= y_axis and y_axis < self.target_map_size:
            self.scan_map[x_axis, y_axis] += amount


