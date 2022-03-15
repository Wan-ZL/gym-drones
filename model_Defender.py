# from or_tool_trajectory import MD_path_plan_main
from or_tools_test import MD_path_plan_main
import numpy as np

class defender_model:
    def __init__(self, system):
        self.system = system        # save the pointer to system class
        self.xyz_axis = [0,0,0]     # Regional Leader Drone location
        self.sg_HD = 5 #     defense strategy, range [0, 10]
        self.min_sg_HD = 1  # minimum signal level defender can choose
        self.max_sg_HD = 10  # maximum signal level defender can choose
        self.tao_lower = 1  # The lower bounds of the number of MDs that HDs can protect simultaneously
        self.tao_upper = 3  # The upper bounds of the number of MDs that HDs can protect simultaneously
        # 'self.MD_trajectory' only contain x and y
        self.z_range_start_MD = 1
        self.z_range_end_MD = 2
        self.MD_trajectory = MD_path_plan_main(self.system.num_MD, self.system.target_map_size, self.system.not_scanned_map())
        self.MD_trajecotry_add_Z()

    def MD_trajecotry_add_Z(self):
        # z_range_start_MD = 1
        # z_range_end_MD = 2
        if self.system.num_MD == 1:
            z_mutiplier = self.z_range_end_MD
        else:
            z_mutiplier = (self.z_range_end_MD - self.z_range_start_MD) / (self.system.num_MD - 1)
        for id in range(self.system.num_MD):
            self.MD_trajectory[id] = np.insert(self.MD_trajectory[id], 2, z_mutiplier * id + self.z_range_start_MD, axis=1)

    def update_location(self, x_axis, y_axis, z_axis):
        self.xyz_axis = (x_axis, y_axis, z_axis)

    def MD_trajectory(self, num_MD, target_map_size):
        return MD_path_plan_main(num_MD, target_map_size)

    def is_calc_trajectory(self):                # return 'True' means new trajectory calculated
        # TODO: create trajectory only for not scanned cells
        if self.system.recalc_trajectory:
            print("Calculating MD trajectory......")
            self.system.recalc_trajectory = False
            self.MD_trajectory =  MD_path_plan_main(self.system.num_MD, self.system.target_map_size, self.system.not_scanned_map())
            self.MD_trajecotry_add_Z()
            return True
        else:
            return False

