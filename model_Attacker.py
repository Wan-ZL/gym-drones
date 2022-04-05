from model_player import player_model
import random
import numpy as np
from collections import defaultdict

class attacker_model(player_model):
    def __init__(self, system):
        player_model.__init__(self, system)
        # randomly set to target area when create
        self.xyz = np.array([system.map_size ,system.map_size,0]) #np.array([random.randrange(1,system.map_cell_number+1), random.randrange(1,system.map_cell_number+1), 0])
        self.obs_sig_dict = defaultdict(int)      # key is drone ID, value is observed signal level
        self.S_target_dict = defaultdict(list)
        self.observe()                  # observe environment and add value to 'obs_sig_dict' and 'S_target_dict'
        self.compromise_record = {}     # key: att_stra (observed signal), value: def_stra (since attacker doesn't know actual signal, we use observed signal)
        self.num_att_stra = 9                    # number of attacker strategy
        self.num_def_stra = 9                    # number o f defender strategy
        self.success_record = np.zeros((system.num_MD + system.num_HD, self.num_def_stra))  # row drone ID, column: def_stra
        self.failure_record = np.zeros((system.num_MD + system.num_HD, self.num_def_stra))
        self.strategy = 9                       # attack threshold range (1,10)
        self.strategy2signal_set = [(-100,-98.1), (-98.1,-96.1), (-96.1,-93.8), (-93.8,-91.1), (-91.1,-87.9), (-87.9,-84.0), (-84.0,-79.0), (-79.0,-72.0), (-72.0,-60), (-60,20)]
        # condition edit in 'def observe()'. It convert signal strength to strategy index
        self.target_set = []
        self.epsilon = 0.5                      # variable used in determine target range
        self.attack_success_prob = 0.0          # attack success rate of each attack on each drone

    def signal2strategy(self, obs_signal):
        conditions = lambda x: {
            x < -100: -1, -100 <= x < -98.1: 0, -98.1 <= x < -96.1: 1, -96.1 <= x < -93.8: 2, -93.8 <= x < -91.1: 3,
            -91.1 <= x < -87.9: 4,
            -87.9 <= x < -84.0: 5, -84.0 <= x < -79.0: 6, -79.0 <= x < -72.0: 7, -72.0 <= x < -60: 8, -60 <= x <= 20: 9,
            20 < x: -1
        }
        return conditions(obs_signal)[True]

    # observation action
    def observe(self):
        self.obs_sig_dict = {}       # key: drone ID, value: observed signal strength
        self.S_target_dict = defaultdict(list)    # key: observed signal level, value: drone class
        # conditions = lambda x: {
        #     x < -100: -1, -100 <= x < -98.1: 0, -98.1 <= x < -96.1: 1, -96.1 <= x < -93.8: 2, -93.8 <= x < -91.1: 3,
        #     -91.1 <= x < -87.9: 4,
        #     -87.9 <= x < -84.0: 5, -84.0 <= x < -79.0: 6, -79.0 <= x < -72.0: 7, -72.0 <= x < -60: 8, -60 <= x <= 20: 9,
        #     20 < x: -1
        # }

        distance_dict = {}
        for MD in self.system.MD_mission_set:   # only consider MD in mission and not crashed
            distance = self.system.calc_distance(self.xyz, MD.xyz_temp)
            distance_dict[MD.ID] = distance
            obs_signal = self.system.observed_signal(MD.signal, distance)
            self.obs_sig_dict[MD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)
            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [MD]
        for HD in self.system.HD_mission_set:         # we consider crashed drone here
            distance = self.system.calc_distance(self.xyz, HD.xyz_temp)
            distance_dict[HD.ID] = distance
            obs_signal = self.system.observed_signal(HD.signal, distance)
            self.obs_sig_dict[HD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)
            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [HD]

        print("attacker observed:", self.obs_sig_dict)
        print("attacker obs distance:", distance_dict)      # TODO: check if distance-signal function are correct

    def impact(self):
        ai = np.ones((self.num_att_stra, self.num_def_stra), dtype=float) / (self.num_att_stra * self.num_def_stra)
        max_set = 0
        print("S_target_dict", self.S_target_dict)
        for att_stra in range(self.num_att_stra):
            # find denominator
            if len(self.S_target_dict[att_stra]) > max_set:
                max_set = len(self.S_target_dict[att_stra])

            # calculate numerator
            for def_stra in range(self.num_def_stra):
                # numerat_sum = 0
                for drone in self.S_target_dict[att_stra]:
                    if self.success_record[drone.ID, def_stra]:
                        ai[att_stra, def_stra] += (self.success_record[drone.ID, def_stra]/ (self.success_record[drone.ID, def_stra] + self.failure_record[drone.ID, def_stra]))

        print("ai", ai)
        print("max_set", max_set)

        ai = ai/max_set
        return ai

    def select_strategy(self):
        # ai = self.impact()
        self.strategy = 9
        pass

    def action(self):
        print("attacker strategy:", self.strategy, "signal", self.strategy2signal_set[self.strategy])
        target_set = self.S_target_dict[self.strategy]
        for drone in target_set:
            print("attacking", drone.ID, drone.type)
            # only attack not crashed drone
            if random.uniform(0,1) < self.attack_success_prob:
                print("attack success:", drone)
                drone.xyz[2] = 0
                drone.xyz_temp[2] = 0
                drone.crashed = True
















