'''
Project     ：gym-drones 
File        ：A3C_train_agent_optuna_5.py
Author      ：Zelin Wan
Date        ：12/2/22
Description : 
'''

import os

os.environ["OMP_NUM_THREADS"] = "1"  # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch.multiprocessing as mp
import numpy as np
import optuna
import random
import time

from shared_adam import SharedAdam
from A3C_model_5 import *
from sys import platform
from multiprocessing import Manager
from utils import get_last_ten_ave
from torch.utils.tensorboard import SummaryWriter
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim


def objective(trial, fixed_seed=True, on_server=True, exist_model=False, exp_scheme=0, player_name='def',
              is_custom_env=True, miss_dur=30, target_size=5):
    start_time = time.time()

    if trial is not None:
        trial_num_str = str(trial.number)
    else:
        trial_num_str = "None"

    # This Configuration May Be Changed By Optuna:
    # glob_episode_thred: total number episode runs,
    # min_episode: minimum number of episode allowed to run before optuna pruning,
    # gamma: used to discount future reward, lr: learning rate, LR_decay: learning rate decay,
    # epsilon: probability of doing random action, epsilon_decay: epsilon decay,
    # pi_net_struc: structure of policy network, v_net_struct: structure of value network.

    # default setting for players
    glob_episode = 5000
    # default setting for defender
    config_def = dict(glob_episode_thred=glob_episode, min_episode=1000, gamma=0.913763771439141,
                      lr=0.00135463670164839,
                      LR_decay=0.97127344458321, epsilon=0.0119790647086112,
                      epsilon_decay=0.970168515121627, pi_net_struc=[64, 128, 128, 64], v_net_struct=[64, 128, 128, 64])

    # default setting for attacker
    config_att = dict(glob_episode_thred=glob_episode, min_episode=1000, gamma=0.988784695574701,
                      lr=0.00861718519746169,
                      LR_decay=0.974819836482024, epsilon=0.0642399295782559,
                      epsilon_decay=0.961601926251759, pi_net_struc=[64, 128, 128, 64], v_net_struct=[64, 128, 128, 64])

    # Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        if exp_scheme == 0 or exp_scheme == 2:  # those two scheme use D-a3c
            # trial for defender
            config_def["gamma"] = trial.suggest_loguniform('gamma', 0.9, 0.99)
            config_def["lr"] = trial.suggest_loguniform('lr', 0.001, 0.01)
            config_def["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.9, 0.99)
            config_def["epsilon"] = trial.suggest_loguniform('epsilon', 0.01, 0.5)
            config_def["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay', 0.9, 0.99)

        if exp_scheme == 1 or exp_scheme == 2:  # those two scheme use A-a3c
            # trial for attacker
            config_att["gamma"] = trial.suggest_loguniform('gamma', 0.9, 0.99)
            config_att["lr"] = trial.suggest_loguniform('lr', 0.001, 0.01)
            config_att["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.9, 0.99)
            config_att["epsilon"] = trial.suggest_loguniform('epsilon', 0.01, 0.5)
            config_att["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay', 0.9, 0.99)

    if exist_model:
        config_def["epsilon"] = 0
        config_att["epsilon"] = 0
    print("config_def", config_def)
    print("config_att", config_att)

    if is_custom_env:
        env = HyperGameSim(fixed_seed=fixed_seed)
        input_dims_def = env.observation_space['def'].shape[0]
        n_actions_def = env.action_space['def'].n
        input_dims_att = env.observation_space['att'].shape[0]
        n_actions_att = env.action_space['att'].n
        env.close_env()
    else:
        env = gym.make('CartPole-v1')
        input_dims_def = env.observation_space.shape[0]
        # input_dims = env.observation_space.n
        n_actions_def = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)

    # defender's global AC
    glob_AC_def = ActorCritic(input_dims=input_dims_def, n_actions=n_actions_def, epsilon=config_def["epsilon"],
                              epsilon_decay=config_def["epsilon_decay"], pi_net_struc=config_def["pi_net_struc"],
                              v_net_struct=config_def["v_net_struct"], fixed_seed=fixed_seed).to(
        device)  # Global Actor Critic
    if exist_model:
        # load pre-trained model's parameters
        path = "trained_model/DefAtt/def/trained_A3C"
        glob_AC_def.load_state_dict(torch.load(path))
        glob_AC_def.eval()
    print("global_actor_critic", glob_AC_def)
    glob_AC_def.share_memory()  # share the global parameters in multiprocessing
    optim_def = SharedAdam(glob_AC_def.parameters(), lr=config_def["lr"])  # global optimizer

    # attacker's global AC
    glob_AC_att = ActorCritic(input_dims=input_dims_att, n_actions=n_actions_att, epsilon=config_att["epsilon"],
                              epsilon_decay=config_att["epsilon_decay"], pi_net_struc=config_att["pi_net_struc"],
                              v_net_struct=config_att["v_net_struct"], fixed_seed=fixed_seed).to(
        device)  # Global Actor Critic
    if exist_model:
        # load pre-trained model's parameters
        path = "trained_model/DefAtt/att/trained_A3C"
        glob_AC_att.load_state_dict(torch.load(path))
        glob_AC_att.eval()
    print("global_actor_critic", glob_AC_att)
    glob_AC_att.share_memory()  # share the global parameters in multiprocessing
    optim_att = SharedAdam(glob_AC_att.parameters(), lr=config_att["lr"])  # global optimizer

    # shared_dict is shared by all parallel processes
    shared_dict = {}
    shared_dict['global_ep'] = mp.Value('i', 0)
    shared_dict['glob_r_list'] = Manager().list()
    shared_dict["start_time"] = str(start_time)
    shared_dict['eps_writer'] = mp.Queue()  # '_writer' means it will be used for tensorboard writer
    shared_dict['score_def_writer'] = mp.Queue()
    shared_dict['score_att_writer'] = mp.Queue()
    shared_dict['score_def_avg_writer'] = mp.Queue()
    shared_dict['score_att_avg_writer'] = mp.Queue()
    shared_dict['def_stra_count_writer'] = mp.Queue()
    shared_dict['att_stra_count_writer'] = mp.Queue()
    shared_dict['lr_writer'] = mp.Queue()
    shared_dict['lr_def_writer'] = mp.Queue()
    shared_dict['lr_att_writer'] = mp.Queue()
    shared_dict["epsilon_writer"] = mp.Queue()
    shared_dict["epsilon_def_writer"] = mp.Queue()
    shared_dict["epsilon_att_writer"] = mp.Queue()
    shared_dict['t_loss_writer'] = mp.Queue()
    shared_dict['t_loss_writer'] = mp.Queue()
    shared_dict['c_loss_writer'] = mp.Queue()
    shared_dict['a_loss_writer'] = mp.Queue()
    shared_dict['t_loss_def_writer'] = mp.Queue()  # total loss of actor critic for defender
    shared_dict['c_loss_def_writer'] = mp.Queue()  # loss of critic for defender
    shared_dict['a_loss_def_writer'] = mp.Queue()  # loss of actor for defender
    shared_dict['t_loss_att_writer'] = mp.Queue()  # total loss of actor critic for attacker
    shared_dict['c_loss_att_writer'] = mp.Queue()  # loss of critic for attacker
    shared_dict['a_loss_att_writer'] = mp.Queue()  # loss of actor for attacker
    shared_dict['att_succ_rate_writer'] = mp.Queue()  # attack success rate
    shared_dict['mission_condition_writer'] = mp.Queue()  # 0 means mission fail, 1 means mission success
    shared_dict['total_energy_consump_writer'] = mp.Queue()  # energy consumption of drones
    shared_dict['scan_percent_writer'] = mp.Queue()  # percentage of scanned cell
    shared_dict['def_reward_0_writer'] = mp.Queue()  # the first component of defender's reward
    shared_dict['def_reward_1_writer'] = mp.Queue()  # the second component of defender's reward
    shared_dict['def_reward_2_writer'] = mp.Queue()  # the third component of defender's reward
    shared_dict['att_reward_0_writer'] = mp.Queue()  # the first component of attacker's reward
    shared_dict['att_reward_1_writer'] = mp.Queue()  # the second component of attacker's reward
    shared_dict['att_reward_2_writer'] = mp.Queue()  # the third component of attacker's reward
    shared_dict['MDHD_active_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()
    shared_dict['mission_complete_rate_writer'] = mp.Queue()
    shared_dict['remaining_time_writer'] = mp.Queue()
    shared_dict['energy_HD_writer'] = mp.Queue()
    shared_dict['energy_MD_writer'] = mp.Queue()
    shared_dict['on_server'] = on_server

    # global_ep_r = mp.Value('d', 0.)

    # parallel training
    if on_server:
        num_worker = 75  # mp.cpu_count()       # update this for matching server's resources (ARC can hold up to 75
        # workers if allocate 128 CPUs)
    else:
        num_worker = 10
    # workers = [Agent(glob_AC_def, optim_def, shared_dict=shared_dict, lr_decay=config_def["LR_decay"], gamma=config_def["gamma"],
    #                  epsilon=config_def["epsilon"], epsilon_decay=config_def["epsilon_decay"],
    #                  MAX_EP=config_def["glob_episode_thred"], fixed_seed=fixed_seed, trial=trial,
    #                  name_id=i, player=player_name, is_custom_env=is_custom_env, miss_dur=miss_dur, target_size=target_size) for i in range(num_worker)]
    workers = [Agent(glob_AC_def, glob_AC_att, optim_def, optim_att, shared_dict=shared_dict, config_def=config_def,
                     config_att=config_att, fixed_seed=fixed_seed, trial=trial,
                     name_id=i, exp_scheme=exp_scheme, player=player_name, exist_model=exist_model,
                     is_custom_env=is_custom_env, miss_dur=miss_dur,
                     target_size=target_size) for i in range(num_worker)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    # ========= Save Data for Optuna =========
    score_list = [ele for ele in shared_dict['glob_r_list']]  # get reward of all local agents
    last_10_per_reward_mean = get_last_ten_ave(score_list)

    # ========= Save global model =========
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    if on_server:
        path = "/home/zelin/Drone/data/" + str(miss_dur) + "_" + str(target_size) + "/" + player_name + "/"
    else:
        path = "/Users/wanzelin/办公/gym-drones/data/" + str(miss_dur) + "_" + str(target_size) + "/" + player_name + "/"
    os.makedirs(path + "model", exist_ok=True)
    torch.save(glob_AC_def.state_dict(),
               path + "model/trained_A3C_" + str(start_time) + "_" + player_name + "_Trial_" + trial_num_str)

    # ========= Write Hparameter to Tensorboard =========
    # convert list in self.config to integers
    temp_config = {}
    for key, value in config_def.items():
        if key == 'pi_net_struc':
            temp_config['pi_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['pi_net' + str(index)] = num_node
        elif key == 'v_net_struct':
            temp_config['v_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['v_net' + str(index)] = num_node
        else:
            temp_config[key] = value
    if on_server:
        path = "/home/zelin/Drone/data/" + str(miss_dur) + "_" + str(target_size) + "/"
    else:
        path = "data/" + str(miss_dur) + "_" + str(target_size) + "/"
    writer_hparam = SummaryWriter(
        log_dir=path + "runs_" + player_name + "/each_run_" + str(start_time) + "-" + player_name +
                "-Trial_" + trial_num_str + "-hparm")

    writer_hparam.add_hparams(temp_config, {'return_reward': last_10_per_reward_mean})  # add for Hyperparameter Tuning
    writer_hparam.flush()
    writer_hparam.close()

    return last_10_per_reward_mean  # return average value


if __name__ == '__main__':
    test_mode = True  # True means use preset hyperparameter, and optuna will not be used.
    exist_model = False  # True means use the existing pre-trained models. False means train new models.
    exp_scheme = 0  # 0 means A-random D-a3c, 1 means A-a3c D-random, 2 means A-a3c D-a3c, 3 means A-random D-random
    # is_defender = True      # True means train a defender RL, False means train an attacker RL
    fixed_seed = True  # True means the seeds for pytorch, numpy, and python will be fixed.
    is_custom_env = True  # True means use the customized drone environment, False means use gym 'CartPole-v1'.

    # sensitivity analysis value
    miss_dur = 30  # default: 30
    target_size = 5  # default: 5. The 'Number of Cell to Scan' = 'target_size' * 'target_size'

    test_mode_run_time = 10

    if exp_scheme == 0:
        player_name = 'def'
        print("running for defender")
    elif exp_scheme == 1:
        player_name = 'att'
        print("running for attacker")
    elif exp_scheme == 2:
        player_name = 'DefAtt'
        print("running for defender and attacker")
    elif exp_scheme == 3:
        player_name = 'random'
        fixed_seed = False
        print("running randomly for defender and attacker")
    else:
        raise Exception("invalid exp_scheme, see comment")

    if fixed_seed:
        # torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    if platform == "darwin":
        on_server = False
    else:
        on_server = True
    # objective(None)
    # 3. Create a study object and optimize the objective function.
    # /home/zelin/Drone/data
    if test_mode:
        print("testing mode")
        for _ in range(test_mode_run_time):
            objective(None, fixed_seed=fixed_seed, on_server=on_server, exist_model=exist_model, exp_scheme=exp_scheme,
                      player_name=player_name, is_custom_env=is_custom_env, miss_dur=miss_dur, target_size=target_size)
    else:
        print("training mode")
        if on_server:
            db_path = "/home/zelin/Drone/data/" + str(miss_dur) + "_" + str(target_size) + "/" + player_name + "/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite://///" + db_path + "HyperPara_database.db",
                                        load_if_exists=True)
        else:
            db_path = "/Users/wanzelin/办公/gym-drones/data/" + str(miss_dur) + "_" + str(
                target_size) + "/" + player_name + "/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite:////" + db_path + "HyperPara_database.db",
                                        load_if_exists=True)
        study.optimize(
            lambda trial: objective(trial, fixed_seed, on_server, exist_model, exp_scheme, player_name, is_custom_env,
                                    miss_dur, target_size), n_trials=100)
