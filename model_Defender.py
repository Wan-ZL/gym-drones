from or_tool_test import MD_path_plan_main


class defender_model:
    def __init__(self, system):
        self.system = system        # save the pointer to system class
        self.xyz_axis = [0,0,0]     # Regional Leader Drone location
        self.sg_HD = 5 #     defense strategy, range [0, 10]
        self.min_sg_HD = 1  # minimum signal level defender can choose
        self.max_sg_HD = 10  # maximum signal level defender can choose
        self.tao_lower = 1  # The lower bounds of the number of MDs that HDs can protect simultaneously
        self.tao_upper = 3  # The upper bounds of the number of MDs that HDs can protect simultaneously

    def update_location(self, x_axis, y_axis, z_axis):
        self.xyz_axis = (x_axis, y_axis, z_axis)

    def MD_trajectory(self, num_MD, target_map_size):
        return MD_path_plan_main(num_MD, target_map_size)

    def create_trajectory(self):
        # TODO: create trajectory only for not scanned cells
        pass

