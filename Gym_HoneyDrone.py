from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym import Env
# from gym.spaces import Discrete, Box
from gym import spaces
import numpy as np

class HoneyDrone(CtrlAviary):
    def __init__(self, min_sig=-100, max_sig = 20, *args, **kwargs):
        self.min_sig = min_sig
        self.max_sig = max_sig
        super(HoneyDrone, self).__init__(*args, **kwargs)   # completely inherit from CtrlAviary including all parameters

        # print("space is ",self.action_space, self.action_space[str(0)].shape)
        # self.action_space = spaces.Discrete(self.strategy_number)
        # quit()


    # modified, add a action for signal strength
    def _actionSpace(self):
        #### Action vector ######## P0            P1            P2            P3        signal_strength
        act_lower_bound = np.array([0.,           0.,           0.,           0.,       self.min_sig])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.max_sig])
        return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})