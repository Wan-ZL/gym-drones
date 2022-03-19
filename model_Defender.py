from or_tool_trajectory import MD_path_plan_main
# from or_tools_test import MD_path_plan_main
from model_player import player_model
import numpy as np

class defender_model(player_model):
    def __init__(self, system):
        player_model.__init__(self, system)
        self.sg_HD = 5 #     defense strategy, range [0, 10]
        self.min_sg_HD = 1  # minimum signal level defender can choose
        self.max_sg_HD = 10  # maximum signal level defender can choose
        self.tao_lower = 1  # The lower bounds of the number of MDs that HDs can protect simultaneously
        self.tao_upper = 3  # The upper bounds of the number of MDs that HDs can protect simultaneously
        # 'self.MD_trajectory' only contain x and y
        self.z_range_start_MD = 1
        self.z_range_end_MD = 2
        self.alive_index_set, self.alive_posi_set = self.MD_position_enumerate()
        self.MD_trajectory = MD_path_plan_main(self.alive_index_set, self.alive_posi_set, self.system.target_map_size, self.system.not_scanned_map())
        self.MD_trajecotry_add_Z()

    # only consider non compromised MD
    def MD_position_enumerate(self):
        res_posi_set = []
        res_index_set = []
        for MD in self.system.MD_set:
            if MD.compromised:
                continue
            xy_list = MD.xyz[:2].astype(int).tolist()
            res_posi_set.append(tuple(xy_list))
            res_index_set.append(MD.ID)
            # res_set.append(MD.xyz[:2].astype(int).tolist())

        return res_index_set, res_posi_set

    def MD_trajectory_remove_head(self):
        for MD in self.system.MD_set:
            if MD.compromised:
                continue

            self.MD_trajectory[MD.ID] = np.delete(self.MD_trajectory[MD.ID], 0, 0)
            # self.MD_trajectory[MD.ID] = self.MD_trajectory[MD.ID][1:]

        # for id in range(self.system.num_MD):
        #     self.MD_trajectory[id] = self.MD_trajectory[id][1:]


    def MD_trajecotry_add_Z(self):
        # z_range_start_MD = 1
        # z_range_end_MD = 2
        if len(self.system.MD_set) == 1:
            z_mutiplier = self.z_range_end_MD
        else:
            z_mutiplier = (self.z_range_end_MD - self.z_range_start_MD) / (len(self.system.MD_set) - 1)
        for MD in self.system.MD_set:
            if MD.compromised:
                continue
            self.MD_trajectory[MD.ID] = np.insert(self.MD_trajectory[MD.ID], 2, z_mutiplier * MD.ID + self.z_range_start_MD, axis=1)
        # for id in range(self.system.num_MD):
        #     self.MD_trajectory[id] = np.insert(self.MD_trajectory[id], 2, z_mutiplier * id + self.z_range_start_MD, axis=1)


    # def MD_trajectory(self, num_MD, target_map_size):
    #     return MD_path_plan_main(num_MD, target_map_size)

    def is_calc_trajectory(self):                # return 'True' means new trajectory calculated
        if self.system.recalc_trajectory and self.system.mission_Not_end == self.system.mission_max_status:
            alive_index_set, alive_posi_set = self.MD_position_enumerate()
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

