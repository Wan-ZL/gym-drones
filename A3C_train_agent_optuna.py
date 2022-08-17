'''
@Project ：gym-drones 
@File    ：A3C_train_agent_optuna.py
@Author  ：Zelin Wan
@Date    ：8/15/22
'''

import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"

import time
import optuna

from sys import platform
from A3C_model import *
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


def objective(trial):
    start_time = time.time()
    is_defender = False     # True means train a defender RL, False means train an attacker RL
    if is_defender:
        player_name = 'def'
    else:
        player_name = 'att'

    if trial is not None:
        trial_num_str = str(trial.number)
    else:
        trial_num_str = "None"
    writer_hparam = SummaryWriter("runs/each_run_" +str(start_time) + "-" + player_name + "-" + "-Trial_" + trial_num_str)

    config = dict(glob_episode_thred=5000, gamma=0.3, lr=0.00020036, LR_decay=0.972, pi_net_struc=[384, 384, 512], v_net_struct=[256])    # this config may be changed by optuna

    # 2. Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        # config["glob_episode_thred"] = trial.suggest_int('glob_episode_thred', 1000, 1500, 100)     # total number of episodes
        config["gamma"] = trial.suggest_loguniform('gamma', 0.9, 1.0)
        config["lr"] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        config["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.9, 1.0)   # since scheduler is not use. This one has no impact to reward
        pi_n_layers = trial.suggest_int('pi_n_layers', 3, 5)  # total number of layer
        config["pi_net_struc"] = []     # Reset before append
        for i in range(pi_n_layers):
            config["pi_net_struc"].append(trial.suggest_int(f'pi_n_units_l{i}', 32, 128, 32))   # try various nodes each layer
        v_n_layers = trial.suggest_int('v_n_layers', 3, 5)  # total number of layer
        config["v_net_struct"] = []     # Reset before append
        for i in range(v_n_layers):
            config["v_net_struct"].append(trial.suggest_int(f'v_n_units_l{i}', 32, 128, 32))  # try various nodes each layer
    print("config", config)

    if on_server:
        num_worker = 125  # mp.cpu_count()     # update this for matching server's resources
    else:
        num_worker = 2

    temp_env = HyperGameSim()
    n_actions = temp_env.action_space.n
    input_dims = temp_env.observation_space.shape
    temp_env.close_env()    # close client for avoiding client limit error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)
    global_actor_critic = ActorCritic(input_dims, n_actions,
                                      gamma=config["gamma"],
                                      pi_net_struc=config["pi_net_struc"],
                                      v_net_struct=config["v_net_struct"]).to(device)  # global NN
    print(global_actor_critic)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=config['lr'],
                       betas=(0.9, 0.999))
    global_ep = mp.Value('i', 0)

    def lambda_function(epoch):  # epoch increase one when scheduler.step() is called
        return config["LR_decay"] ** epoch

    scheduler = LambdaLR(optim, lr_lambda=lambda_function)
    # scheduler = None    # don't use scheduler



    shared_dict = {}
    shared_dict["reward"] = Manager().list()  # use Manager().list() to create a shared list between processes
    shared_dict["t_step"] = Manager().list()
    shared_dict["att_action"] = mp.Array('i', n_actions)
    shared_dict["start_time"] = str(start_time)
    shared_dict["ave_10_per_return"] = Manager().list()
    # shared_dict["reward"] = mp.Array('d', glob_episode_thred + num_worker)        # save simulation data ('d' means double-type)

    workers = [Agent(global_actor_critic,
                     optim,
                     scheduler,
                     input_dims,
                     n_actions,
                     name=i,
                     global_ep_idx=global_ep,
                     global_dict=shared_dict,
                     config=config,
                     glob_episode_thred=config['glob_episode_thred'],
                     player="att") for i in range(num_worker)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    global_reward_10_per = [ele for ele in shared_dict["ave_10_per_return"]]    # get reward of all local agents
    ave_global_reward_10_per = sum(global_reward_10_per)/len(global_reward_10_per)

    # ========= Save global model =========
    if on_server:
        path = "/home/zelin/Drone/code_files/data/attacker"
    else:
        path = "/Users/wanzelin/办公/gym-drones/data/A3C/defender"
    os.makedirs(path + "/model", exist_ok=True)
    torch.save(global_actor_critic, path + "/model/trained_A3C_attacker_" + str(start_time))

    # Write hparameter to tensorboard
    # convert list in self.config to integers
    temp_config = {}
    for key, value in config.items():
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
    writer_hparam.add_hparams(temp_config, {'return_reward': ave_global_reward_10_per})  # add for Hyperparameter Tuning
    writer_hparam.flush()
    writer_hparam.close()

    return ave_global_reward_10_per  # return average value


if __name__ == '__main__':
    if platform == "darwin":
        on_server = False
    else:
        on_server = True
    # objective(None)
    # 3. Create a study object and optimize the objective function.
    # /home/zelin/Drone/data
    if on_server:
        study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                    storage="sqlite://////home/zelin/Drone/code_files/data/attacker/HyperPara_database.db",
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                    storage="sqlite://///Users/wanzelin/办公/gym-drones/data/attacker/HyperPara_database.db",
                                    load_if_exists=True)
    study.optimize(objective, n_trials=100)
