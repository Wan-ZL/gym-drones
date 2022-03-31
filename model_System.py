import numpy as np
from model_MD import Mission_Drone
from model_HD import Honey_Drone


class system_model:
    def __init__(self):
        self.update_freq = 500  # environment frame per round
        self.mission_Not_end = 2  # 2 means True, use 2 here to allow drone back to BaseStation in debug mode
        self.mission_success = False
        self.mission_max_status = self.mission_Not_end
        self.mission_duration = 0  # T_M in paper
        self.mission_max_duration = 200  # T^{max}_m in paper
        self.map_cell_number = 4    # number of cell each side
        self.cell_size = 10  # in meter     (left bottom point of cell represents the coordinate of cell
        self.map_size = self.map_cell_number * self.cell_size   # in meter
        self.map_ori_x = 1  # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.map_size, self.map_size))
        self.min_scan_requirement = 1
        self.recalc_trajectory = False  # True: need recalculate trajectory
        self.num_MD = 10  # number of MD (in index, MD first then HD)
        self.num_HD = 1  # number of HD
        self.MD_dict = {}  # key is id, value is class detail
        self.HD_dict = {}
        self.MD_crashed_IDs = []  # includes MD crashed or crashed
        self.assign_HD()
        self.assign_MD()
        # self.MD_set = self.get_alive_MD()
        # self.HD_set = self.get_alive_HD()

    # distance between two drones. (Eq. \eqref(Eq: distance) in paper)
    def drones_distance(self, drone_x_location, drone_y_location):
        distance_squre = np.square(drone_x_location - drone_y_location)
        return np.sqrt(np.sum(distance_squre))

    # iterate through all MD/HD and check their state (only consider alive drone)
    def Drone_state_update(self):
        for MD in self.MD_set:
            MD.condition_check()
        for HD in self.HD_set:
            HD.condition_check()

    # input type: np.ndarray
    def calc_distance(self, xyz1: np.ndarray, xyz2: np.ndarray):
        return np.linalg.norm(xyz1 - xyz2)

    # return integer as signal, round down decimal points
    # if value greater than 10, return 10
    def observed_signal(self, original_signal, distance):
        res = original_signal - 4 * 10 * np.log10(distance)
        return res

    # the maximum range of a drone under given transmitted signal strength
    # -100 dBm usually treated as the minimum valid signal strength.
    def signal_range(self, original_signal):
        res = 10 ** ((original_signal + 100) / 40)
        return res

    # MD_set only contain MD that not crash
    @property
    def MD_set(self):
        return [MD for MD in self.MD_dict.values() if not MD.crashed]

    # HD_set only contain MD that not crash
    @property
    def HD_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed]

    # scan count, and check if crashed
    def MD_environment_interaction(self, obs, scan_map):
        for MD in self.MD_set:
            if MD.crashed:
                continue

            cell_x, cell_y, height_z = obs[str(MD.ID)]["state"][0:3]

            # check if new drone crashed
            if MD.new_crash(height_z):
                self.recalc_trajectory = True   # when a MD offline, recalculate trajectory

            # update scanned map
            map_x_index = int(cell_x + 0.5)
            map_y_index = int(cell_y + 0.5)
            map_size_with_station = self.map_size + 1

            if 0 <= map_x_index and map_x_index < map_size_with_station and 0 <= map_y_index and map_y_index < map_size_with_station:
                scan_map[map_x_index, map_y_index] += 0.01
                self.update_scan(map_x_index, map_y_index, 0.01)
            else:
                print("out of Mape range")
                print("ID, cell_x, cell_y, height_z", MD.ID, cell_x, cell_y, height_z)
                print("map_x_index_const, map_y_index_const", MD.ID, map_x_index, map_y_index)

    def HD_environment_interaction(self, obs):
        for HD in self.HD_set:
            if HD.crashed:
                continue

            cell_x, cell_y, height_z = obs[str(HD.ID)]["state"][0:3]

            # check if new drone crashed
            if HD.new_crash(height_z):      # when a HD crashed, no trajectory recalculate need
                pass


    def not_scanned_map(self):
        return self.scan_map < self.min_scan_requirement

    def print_MDs(self):
        for MD in self.MD_set:
            print(MD)

    def print_HDs(self):
        for HD in self.HD_set:
            print(HD)

    def print_system(self):
        print(vars(self))

    def print_drones_battery(self):
        for MD in self.MD_set:
            print(f"MD {MD.ID} battery: {MD.battery}")
        for HD in self.HD_set:
            print(f"HD {HD.ID} battery: {HD.battery}")

    def location_backup_MD(self):

        pass

    def battery_consume(self):
        # TODO: avoid recalc when unused drone in base station. Only recalc when: 1. drone crashed. 2. drone low battery. 3.....
        for MD in self.MD_set:
            MD.battery_update()

        for HD in self.HD_set:  # True means battery charging complete
            HD.battery_update()

    def assign_HD(self):
        for index in range(self.num_HD):
            self.HD_dict[index] = Honey_Drone(index, self.update_freq)

    def assign_MD(self):
        for index in range(self.num_MD):
            self.MD_dict[index + self.num_HD] = Mission_Drone(index + self.num_HD, self.update_freq)

    def sample_MD(self):
        return Mission_Drone(-1, self.update_freq)

    @property
    def sample_HD(self):
        return Honey_Drone(-2, self.update_freq)

    # this is system information update function called once per round
    def check_mission_complete(self):
        # update system information
        self.mission_duration += 1

        # check if mission can end
        if np.all(self.scan_map > 1):
            # mission end if: all cells are scanned
            self.mission_Not_end -= 1
            self.mission_success = True
        elif self.mission_duration > self.mission_max_duration:
            # mission end if: mission time limit meet
            self.mission_Not_end -= 1
        else:
            self.print_drones_battery()

        # check if mission end successfully
        if not self.mission_Not_end:
            if self.mission_success:
                print("Mission Complete!!")
                print("scanned map:\n", self.scan_map)
                self.print_MDs()
                self.print_HDs()
                self.print_system()
            else:
                print("Mission Fail :(")
                print("scanned map:\n", self.scan_map)
                self.print_MDs()
                self.print_HDs()
                self.print_system()

        return self.mission_Not_end

    # this is a fast way to check for saving running time
    def is_mission_Not_end(self):
        return self.mission_Not_end

    def update_scan(self, x_axis, y_axis, amount):
        x_axis = x_axis - self.map_ori_x
        y_axis = y_axis - self.map_ori_y
        if 0 <= x_axis and x_axis < self.map_size and 0 <= y_axis and y_axis < self.map_size:
            self.scan_map[x_axis, y_axis] += amount
