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
        self.map_cell_number = 5    # number of cell each side
        self.cell_size = 100  # in meter     (left bottom point of cell represents the coordinate of cell
        self.map_size = self.map_cell_number * self.cell_size   # in meter
        self.map_ori_x = 1  # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.map_size, self.map_size))
        self.scan_cell_map = self.scan_map[::self.cell_size, ::self.cell_size]
        self.min_scan_requirement = 5
        self.recalc_trajectory = False  # True: need recalculate trajectory. TODO: Recalc when charging to full.
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
        if self.mission_duration > 1:
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

    # set of MD in mission (not in GCS)
    @property
    def MD_mission_set(self):
        return [MD for MD in self.MD_dict.values() if not MD.crashed and not MD.in_GCS]

    # HD_set only contain MD that not crash
    @property
    def HD_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed]

    # set of HD in mission (not in GCS)
    @property
    def HD_mission_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed and not HD.in_GCS]

    # scan count, and check if crashed
    def MD_environment_interaction(self, obs):
        for MD in self.MD_set:
            if MD.crashed:
                continue

            xyz_current = obs[str(MD.ID)]["state"][0:3]
            cell_x, cell_y, height_z = xyz_current
            # if new drone crashed，recalculate trajectory
            if MD.new_crash(xyz_current):
                self.recalc_trajectory = True   # when a MD offline, recalculate trajectory

            # update scanned map
            map_x_index = round(cell_x - 1)     # minus 1 for ignoring GCS
            map_y_index = round(cell_y - 1)     # minus 1 for ignoring GCS
            # map_size_with_station = self.map_size + 1


            if map_x_index in range(self.map_size) and map_y_index in range(self.map_size):
                self.scan_map[map_x_index, map_y_index] += 0.01
                # self.update_scan(map_x_index, map_y_index, 0.01)
            # else:
            #     print("out of Mape range")
            #     print("ID, cell_x, cell_y, height_z", MD.ID, xyz_current)
            #     print("map_x_index_const, map_y_index_const", MD.ID, map_x_index, map_y_index)

    def HD_environment_interaction(self, obs):
        for HD in self.HD_set:
            if HD.crashed:
                continue

            xyz_current = obs[str(HD.ID)]["state"][0:3]

            # check if new drone crashed
            if HD.new_crash(xyz_current):      # when a HD crashed, no trajectory recalculate need
                pass


    def not_scanned_map(self):
        return self.scan_map < self.min_scan_requirement

    def print_MDs(self):
        for MD in self.MD_dict.values():
            print(MD)

    def print_HDs(self):
        for HD in self.HD_dict.values():
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
            if MD.battery_update():
                print(f"detected MD {MD.ID} charging complete, recalculate trajectory")
                self.recalc_trajectory = True

        for HD in self.HD_set:  # True means battery charging complete
            if HD.battery_update():
                print(f"detected HD {HD.ID} charging complete, recalculate trajectory")
                self.recalc_trajectory = True

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
        scan_cell_map = self.scan_map[::self.cell_size, ::self.cell_size]   # transfer meter-based scan map to cell-based scan map
        if np.all(scan_cell_map > 1):
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
                print("\n Mission Complete!! \n")
                print("cell-based scanned map:\n", scan_cell_map)
                self.print_MDs()
                self.print_HDs()
                self.print_system()
            else:
                print("\n Mission Fail :( \n")
                print("cell-based scanned map:\n", scan_cell_map)
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
