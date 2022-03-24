from or_tool_trajectory import MD_path_plan_main
# from or_tools_test import MD_path_plan_main
from model_player import player_model
import numpy as np

class defender_model(player_model):
    def __init__(self, system):
        player_model.__init__(self, system)
        self.strategy = 10  # defense strategy, range [1, 10]
        self.tao_lower = 2  # The lower bounds of the number of MDs that HDs can protect simultaneously
        self.tao_upper = 4  # The upper bounds of the number of MDs that HDs can protect simultaneously
        self.z_range_start_MD = 2.2
        self.z_range_end_MD = 3.2
        self.z_interval = 0.2       # determine the height interval between drones
        self.z_list = self.generate_z_list()  # index is drone ID, value is z-axis value
        self.alive_index_set, self.alive_posi_set = self.MD_position_enumerate()
        self.MD_trajectory = MD_path_plan_main(self.alive_index_set, self.alive_posi_set, self.system.target_map_size, self.system.not_scanned_map())
        self.MD_trajecotry_add_Z()
        self.HD_locations = self.assign_HD_locations()
        self.pre_alive_MD_num = self.system.num_MD


    def generate_z_list(self):
        res = np.zeros(self.system.num_MD + self.system.num_HD)

        for id in range(self.system.num_MD + self.system.num_HD):
            res[id] = 1 + id * self.z_interval
        print(res)
        return res


    # only consider non crashed MD
    def MD_position_enumerate(self):
        res_posi_set = []
        res_index_set = []
        for MD in self.system.MD_set:
            if MD.crashed:
                continue
            xy_list = MD.xyz[:2].astype(int).tolist()
            res_posi_set.append(tuple(xy_list))
            res_index_set.append(MD.ID)
            # res_set.append(MD.xyz[:2].astype(int).tolist())

        return res_index_set, res_posi_set

    def MD_trajectory_remove_head(self):
        for MD in self.system.MD_set:
            if MD.crashed:
                continue

            self.MD_trajectory[MD.ID] = np.delete(self.MD_trajectory[MD.ID], 0, 0)
            # self.MD_trajectory[MD.ID] = self.MD_trajectory[MD.ID][1:]

        # for id in range(self.system.num_MD):
        #     self.MD_trajectory[id] = self.MD_trajectory[id][1:]
    def assign_HD_locations(self):
        HD_locations = {}
        for HD in self.system.HD_set:
            HD_locations[HD.ID] = np.zeros(3)
            HD_locations[HD.ID][2] = self.z_list[HD.ID] # np.insert(HD_locations[HD.ID], self.z_list[HD.ID], axis=1)
        return HD_locations

    def MD_trajecotry_add_Z(self):
        for MD in self.system.MD_set:
            self.MD_trajectory[MD.ID] = np.insert(self.MD_trajectory[MD.ID], 2, self.z_list[MD.ID], axis=1)

        # if len(self.system.MD_set) == 1:
        #     z_mutiplier = self.z_range_end_MD
        # else:
        #     z_mutiplier = (self.z_range_end_MD - self.z_range_start_MD) / (len(self.system.MD_set) - 1)
        # for MD in self.system.MD_set:
        #     if MD.crashed:
        #         continue
        #     self.MD_trajectory[MD.ID] = np.insert(self.MD_trajectory[MD.ID], 2, z_mutiplier * MD.ID + self.z_range_start_MD, axis=1)
        # for id in range(self.system.num_MD):
        #     self.MD_trajectory[id] = np.insert(self.MD_trajectory[id], 2, z_mutiplier * id + self.z_range_start_MD, axis=1)


    # def MD_trajectory(self, num_MD, target_map_size):
    #     return MD_path_plan_main(num_MD, target_map_size)

    def is_calc_trajectory(self):                # return 'True' means new trajectory calculated
        # scan all MDs
        alive_index_set, alive_posi_set = self.MD_position_enumerate()

        # check if no alive MD
        if len(alive_index_set) == 0:
            print("No alive MD, Mission Fail")
            self.system.mission_Not_end -= 2
        # check if new drone compromised
        elif len(alive_index_set) < self.pre_alive_MD_num:
            print("detected MD compromised")
            self.system.recalc_trajectory = True
            self.pre_alive_MD_num = len(alive_index_set)

        if self.system.recalc_trajectory and self.system.mission_Not_end == self.system.mission_max_status:
            if len(alive_index_set) == 0:      # if no MD alive, mission fail
                print("No Alive Mission Drone......")
                self.system.mission_Not_end -= 2
                print("Mission Fail :(")
                print("scanned map:\n", self.system.scan_map)
                self.system.print_MDs()
                self.system.print_HDs()
                return False
            else:
                print("Calculating MD trajectory......")
                self.MD_trajectory = MD_path_plan_main(alive_index_set, alive_posi_set, self.system.target_map_size, self.system.not_scanned_map())
                self.MD_trajectory_remove_head()
                self.system.recalc_trajectory = False
                # print("MD_trajectory", self.MD_trajectory)
                # quit()
                self.MD_trajecotry_add_Z()
                return True
        else:
            return False

