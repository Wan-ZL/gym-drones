from or_tool_trajectory import MD_path_plan_main
# from or_tools_test import MD_path_plan_main
from model_player import player_model
import numpy as np

class defender_model(player_model):
    def __init__(self, system):
        player_model.__init__(self, system)
        self.strategy = 10  # defense strategy, range [1, 10]
        self.strategy2signal_set = [-20, -7.9, -0.9, 4.0, 7.9, 11.1, 13.8, 16.1, 18.1, 20]
        self.rho = 5.0        # rho used by sg_{MD} = sg_{HD} - \rho
        self.tao_lower = 2  # The lower bounds of the number of MDs that HDs can protect simultaneously
        self.tao_upper = 4  # The upper bounds of the number of MDs that HDs can protect simultaneously
        self.z_range_start_MD = 2.2
        self.z_range_end_MD = 3.2
        self.z_interval = 0.2       # determine the height interval between drones
        self.z_list = self.generate_z_list()  # index is drone ID, value is z-axis value
        self.alive_index_set, self.alive_posi_set = self.MD_position_enumerate()
        self.MD_trajectory = MD_path_plan_main(self.alive_index_set, self.alive_posi_set, self.system.map_cell_number, self.system.cell_size, self.system.not_scanned_map())
        self.MD_trajecotry_add_Z()
        self.HD_locations = self.assign_HD_locations()
        self.pre_alive_MD_num = self.system.num_MD
        # since we make HD move before defender select strategy, we use DS=5 as default strategy that control P^H_r
        self.p_H_r = self.system.signal_range(self.strategy2signal_set[5])
        self.MD_requirement = 0     # TODO: track how many MD is required in mission

    def select_strategy(self):
        self.strategy = 0

    def action(self):
        HD_signal = self.strategy2signal_set[self.strategy]     # mapping strategy id to signal strength
        HD_signal_radius = self.system.signal_range(HD_signal)
        for HD in self.system.HD_set:
            HD.signal = HD_signal
            HD.signal_radius = HD_signal_radius
        MD_signal = HD_signal - self.rho
        MD_signal_radius = self.system.signal_range(MD_signal)
        for MD in self.system.MD_set:
            MD.signal = MD_signal
            MD.signal_radius = MD_signal_radius
        print("defender strategy:", self.strategy, "HD_signal", HD_signal, "HD_signal_radius", HD_signal_radius)

    # for MD update next destination
    def update_MD_next_destination(self):
        for MD in self.system.MD_set:
            if MD.crashed:
                continue
            MD.assign_destination(self.MD_trajectory[MD.ID][0])
            # MD.xyz = MD_trajectory[MD.ID][0]
            if self.MD_trajectory[MD.ID].shape[0] > 1:
                self.MD_trajectory[MD.ID] = self.MD_trajectory[MD.ID][1:, :]
            else:
                print("MD arrived:", MD.ID)

    # for HD update next destination
    # use Algorithm 1 in paper
    def update_HD_next_destination(self):
        if not self.system.HD_set:      # if no alive HD, ignore the rest
            return

        L_MD_set = np.array([MD.ID for MD in self.system.MD_set])
        # L_HD_set = np.arange(num_MD, num_MD+num_HD)
        L_HD_set = np.arange(self.system.num_HD)

        HD_sample = self.system.HD_set[0]
        max_radius = HD_sample.signal_radius
        # p_H_r = (self.strategy * max_radius) / self.max_strategy  # actual signal radius under given defense strategy sg_HD
        p_H_r = self.p_H_r #HD_sample.signal_radius         # actual signal radius under given defense strategy sg_HD
        S_set_HD = {}  # A set of HDs with assigned MDs
        for HD in self.system.HD_set:
            if L_MD_set.size == 0:
                S_set_HD[HD.ID] = np.empty(0, dtype=int)
                continue

            N_l_H_set = np.empty(0)  # A set of MDs detected/protected by HD
            for MD_id in L_MD_set:
                # if drones_distance(Desti_XYZ[MD_id], Desti_XYZ[HD.ID]) < p_H_r:
                if self.system.drones_distance(self.system.MD_dict[MD_id].xyz, self.system.HD_dict[HD.ID].xyz) < p_H_r:
                    N_l_H_set = np.append(N_l_H_set, MD_id)

            if N_l_H_set.size < self.tao_lower:
                HD_pos_candidate = np.zeros(3)
                N_l_H_new_set = np.empty(0)
                for MD_id_candi in L_MD_set:  # search MD position that HD can move to so more MD can be protected
                    temp_set = np.empty(0)
                    for MD_id in L_MD_set:
                        temp_position = self.system.MD_dict[MD_id].xyz
                        temp_position[:2] = self.system.MD_dict[MD_id].xyz[:2]
                        # if drones_distance(Desti_XYZ[MD_id_candi], Desti_XYZ[MD_id]) < p_H_r:
                        if self.system.drones_distance(self.system.MD_dict[MD_id_candi].xyz, self.system.MD_dict[MD_id].xyz) < p_H_r:
                            temp_set = np.append(temp_set, MD_id)
                    if temp_set.size > N_l_H_new_set.size:  # set new position as candidate
                        N_l_H_new_set = temp_set
                        HD_pos_candidate = self.system.MD_dict[MD_id_candi].xyz
                self.system.HD_dict[HD.ID].assign_destination_xy(HD_pos_candidate[:2])  # nwe position for HD
                # HD_dict[HD.ID].xyz[:2] = HD_pos_candidate[:2]
                N_l_H_new_subset = N_l_H_new_set[:self.tao_upper]
                L_MD_set = np.delete(L_MD_set, np.searchsorted(L_MD_set,
                                                               N_l_H_new_subset))  # Remove protected MDs from set L_MD_set
                S_set_HD[HD.ID] = N_l_H_new_subset  # Add deployed HD to set S_set_HD
            elif self.tao_lower <= N_l_H_set.size and N_l_H_set.size <= self.tao_upper:
                L_MD_set = np.delete(L_MD_set,
                                     np.searchsorted(L_MD_set, N_l_H_set))  # Remove protected MDs from set L_MD_set
                S_set_HD[HD.ID] = N_l_H_set  # Add deployed HD to set S_set_HD
            else:
                N_l_H_subset = N_l_H_set[:self.tao_upper]
                L_MD_set = np.delete(L_MD_set,
                                     np.searchsorted(L_MD_set, N_l_H_subset))  # Remove protected MDs from set L_MD_set
                S_set_HD[HD.ID] = N_l_H_subset  # Add deployed HD to set S_set_HD
            # print("L_MD_set", L_MD_set)
        print("HD protecting", S_set_HD)



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
            HD_locations[HD.ID] = np.ones(3) * int(self.system.map_cell_number) / 2
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


    # def MD_trajectory(self, num_MD, map_cell_number):
    #     return MD_path_plan_main(num_MD, map_cell_number)

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
                self.MD_trajectory = MD_path_plan_main(alive_index_set, alive_posi_set, self.system.map_cell_number, self.system.cell_size, self.system.not_scanned_map())
                self.MD_trajectory_remove_head()
                self.system.recalc_trajectory = False
                # print("MD_trajectory", self.MD_trajectory)
                # quit()
                self.MD_trajecotry_add_Z()
                return True
        else:
            return False

