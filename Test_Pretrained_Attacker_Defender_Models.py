# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C


# TODO: add two models to environment


import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"

import ctypes

import pickle
import time
from copy import copy


import gym
import torch as torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from A3C_model import ActorCritic

# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
#                                          weight_decay=weight_decay)
#
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()
#
#
# class ActorCritic(nn.Module):
#     def __init__(self, input_dims, n_actions, gamma=0.99, pi_net_struc=[128], v_net_struct=[128]):
#         super(ActorCritic, self).__init__()
#
#         self.gamma = gamma
#         self.pi_net = self.build_Net(*input_dims, n_actions, pi_net_struc)
#         self.v_net = self.build_Net(*input_dims, 1, v_net_struct)
#
#         # self.pi1 = nn.Linear(*input_dims, 128)
#         # self.v1 = nn.Linear(*input_dims, 128)
#         # self.pi2 = nn.Linear(128, 256)
#         # self.v2 = nn.Linear(128, 256)
#         # self.pi3 = nn.Linear(256, 128)
#         # self.v3 = nn.Linear(256, 128)
#         # self.pi = nn.Linear(128, n_actions)
#         # self.v = nn.Linear(128, 1)
#
#         self.rewards = []
#         self.actions = []
#         self.states = []
#
#
#     def build_Net(self, obser_space, action_space, net_struc):
#         layers = []
#         in_features = obser_space
#         # for i in range(n_layers):
#         for node_num in net_struc:
#             layers.append(nn.Linear(in_features, node_num))
#             layers.append(nn.ReLU())
#             in_features = node_num
#         layers.append(nn.Linear(in_features, action_space))
#         net = nn.Sequential(*layers)
#         return net
#
#     def remember(self, state, action, reward):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#
#     def clear_memory(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#
#     # def forward(self, state):
#     #     pi1 = F.relu(self.pi1(state))
#     #     v1 = F.relu(self.v1(state))
#     #
#     #     pi2 = self.pi2(pi1)
#     #     v2 = self.v2(v1)
#     #
#     #     pi3 = F.relu(self.pi3(pi2))
#     #     v3 = F.relu(self.v3(v2))
#     #
#     #     pi = self.pi(pi3)
#     #     v = self.v(v3)
#     #
#     #     return pi, v
#
#     def calc_R(self, done):
#         states = torch.tensor(self.states, dtype=torch.float)
#         # _, v = self.forward(states)
#         v = self.v_net(states)
#
#         R = v[-1] * (1 - int(done))
#
#         batch_return = []
#         for reward in self.rewards[::-1]:
#             R = reward + self.gamma * R
#             batch_return.append(R)
#         batch_return.reverse()
#         batch_return = torch.tensor(batch_return, dtype=torch.float)
#
#         return batch_return
#
#     def calc_loss(self, done):
#         states = torch.tensor(self.states, dtype=torch.float)
#         actions = torch.tensor(self.actions, dtype=torch.float)
#
#         returns = self.calc_R(done)
#
#         # pi, values = self.forward(states)
#         pi = self.pi_net(states)
#         values = self.v_net(states)
#
#         values = values.squeeze()
#         critic_loss = (returns - values) ** 2
#
#         probs = torch.softmax(pi, dim=1)
#         dist = Categorical(probs)
#         log_probs = dist.log_prob(actions)
#         actor_loss = -log_probs * (returns - values)
#
#         total_loss = (critic_loss + actor_loss).mean()
#
#         return total_loss
#
#     def choose_action(self, observation):
#         state = torch.tensor([observation], dtype=torch.float)
#         # pi, v = self.forward(state)
#         pi = self.pi_net(state)
#         # v = self.v_net(state)
#         probs = torch.softmax(pi, dim=1)
#         dist = Categorical(probs)
#         action = dist.sample().numpy()[0]
#
#         return action
#
#
# class Agent(mp.Process):
#     def __init__(self, global_actor_critic, optimizer, scheduler, input_dims, n_actions,
#                  name, global_ep_idx, glob_episode_thred, global_dict, config):
#         super(Agent, self).__init__()
#         self.env = HyperGameSim()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Local Using", self.device)
#         self.local_actor_critic = ActorCritic(input_dims, n_actions,
#                                               gamma=config["gamma"], pi_net_struc=config["pi_net_struc"],
#                                               v_net_struct=config["v_net_struct"]).to(self.device)
#         self.global_actor_critic = global_actor_critic
#         self.name = 'w%02i' % name
#         print("creating: " + self.name)
#         self.episode_idx = global_ep_idx
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.glob_episode_thred = glob_episode_thred
#         self.shared_dict = global_dict
#         self.config = config
#
#         # self.writer = writer
#
#     def run(self):
#         # create writer for TensorBoard
#         # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
#         writer = SummaryWriter("runs_att/each_run_" + self.shared_dict["start_time"])
#
#         while self.episode_idx.value < self.glob_episode_thred:
#             # Episode start
#             t_step = 1
#             done = False
#             observation = self.env.reset()
#             score = 0
#             self.local_actor_critic.clear_memory()
#             while not done:
#                 action = self.local_actor_critic.choose_action(observation)
#                 self.shared_dict["att_action"][action] += 1
#                 observation_, reward_def, reward_att, done, info = self.env.step(action_att=action)
#                 score += reward_att
#                 self.local_actor_critic.remember(observation, action, reward_att)
#                 if t_step % 5 == 0 or done:
#                     loss = self.local_actor_critic.calc_loss(done)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     for local_param, global_param in zip(
#                             self.local_actor_critic.parameters(),
#                             self.global_actor_critic.parameters()):
#                         global_param._grad = local_param.grad
#                     self.optimizer.step()
#                     self.local_actor_critic.load_state_dict(
#                         self.global_actor_critic.state_dict())
#                     self.local_actor_critic.clear_memory()
#                 t_step += 1
#                 observation = observation_
#
#             with self.episode_idx.get_lock():
#                 self.episode_idx.value += 1
#
#             # this one commented out for save memory, uncomment as needed.
#             # save data
#             self.shared_dict["reward"].append(score)
#             # self.shared_dict["t_step"].append(t_step)       # number of round for a game
#
#             # self.scheduler.step()  # update learning rate each episode
#
#             print(self.name, 'global-episode ', self.episode_idx.value, 'reward %.1f' % score)
#             writer.add_scalar("Score", score, self.episode_idx.value)
#
#
#         global_reward = [ele for ele in self.shared_dict["reward"]]  # the return of the last 10 percent episodes
#         len_last_return = max(1, int(len(global_reward) * 0.1))     # max can make sure at lead one element in list
#         last_ten_percent_return = global_reward[-len_last_return:]
#
#         ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
#
#         self.shared_dict["ave_10_per_return"].append(ave_10_per_return)
#
#         writer.flush()
#         writer.close()  # close SummaryWriter of TensorBoard
#         return


def att_def_interaction(defender_model, attacker_model):
    start_time = time.time()
    # run command "tensorboard - -logdir = runs_att_def_interaction"
    scenario = 'def'    # 'random', or 'att, or 'def', or 'att_def'
    writer = SummaryWriter("runs_att_def_interaction/each_run_" +str(start_time) + '_' + scenario)

    config = dict(episode=100)    # this config may be changed by optuna

    print("config", config)
    print("attacker_model", attacker_model)
    print("defender_model", defender_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    env = HyperGameSim()

    for episode in range(config['episode']):
        # Episode start
        step_counter = 0
        done = False
        observation = env.reset()
        score_def = 0
        score_att = 0
        while not done:
            action_def = defender_model.choose_action(observation)
            # print("action_def", action_def)
            action_att = attacker_model.choose_action(observation)
            # print("action_att", action_att)
            # observation, reward_def, reward_att, done, info = env.step(action_def=action_def, action_att=action_att)
            if scenario == 'att':
                observation, reward_def, reward_att, done, info = env.step(action_att=action_att)
            elif scenario == 'def':
                observation, reward_def, reward_att, done, info = env.step(action_def=action_def)
            elif scenario == 'att_def':
                observation, reward_def, reward_att, done, info = env.step(action_def=action_def, action_att=action_att)
            else:
                observation, reward_def, reward_att, done, info = env.step()
            score_def += reward_def
            score_att += reward_att
            step_counter += 1
        score_def_average = score_def/step_counter
        score_att_average = score_att/ step_counter
        print('global-episode ', episode, 'score_def_average %.1f' % score_def_average, 'score_att_average %.1f' % score_att_average)
        writer.add_scalar("Defender average Score", score_def_average, episode)
        writer.add_scalar("Attacker average Score", score_att_average, episode)
        writer.add_scalar("Mission Time (step)", step_counter, episode)
        # battery consumpiton for all drones
        consumption_all = 0
        for MD in env.system.MD_dict.values():
            consumption_all += MD.accumulated_consumption
        for HD in env.system.HD_dict.values():
            consumption_all += HD.accumulated_consumption
        writer.add_scalar("Energy Consumption", consumption_all, episode)
        if env.attacker.att_counter == 0:
            att_succ_rate = 0
        else:
            att_succ_rate = env.attacker.att_succ_counter/env.attacker.att_counter
        writer.add_scalar("Attack Success Rate", att_succ_rate, episode)
        writer.add_scalar("Mission Success Rate", env.system.scanCompletePercent(), episode)
    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))
    writer.flush()
    writer.close()  # close SummaryWriter of TensorBoard


if __name__ == '__main__':
    # def_model_path = "/Users/wanzelin/办公/gym-drones/data_for_defender/model/trained_A3C_defender_1658376349.2343273-Trial_21"
    def_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_model_drl_vs_drl/trained_A3C_defender_v2"
    defender_model = torch.load(def_model_path)
    defender_model.eval()
    # att_model_path = "/Users/wanzelin/办公/gym-drones/data_for_attacker/model/trained_A3C_attacker_1658385589.836977"
    att_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_model_drl_vs_drl/trained_A3C_attacker_v2"
    attacker_model = torch.load(att_model_path)
    attacker_model.eval()
    att_def_interaction(defender_model, attacker_model)
    # 3. Create a study object and optimize the objective function.
    # /home/zelin/Drone/data
    # # study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study", storage="sqlite://///Users/wanzelin/办公/gym-drones/data/attacker/HyperPara_database.db", load_if_exists=True)
    # study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study", storage="sqlite://////home/zelin/Drone/code_files/data/attacker/HyperPara_database.db", load_if_exists=True)
    # study.optimize(objective, n_trials=100)

    # Saving data to file
    # Reward
    # global_reward = [ele for ele in shared_dict["reward"]]  # convert mp.Array to python list
    # print(f"reward {global_reward}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/reward_train_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_reward, the_file)
    # the_file.close()
    #
    # # t_step (number of round in a game)
    # global_t_step = [ele for ele in shared_dict["t_step"]]
    # print(f"t_step {global_t_step}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/t_step_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_t_step, the_file)
    # the_file.close()
    #
    # # defender action frequency
    # global_def_action = [ele for ele in shared_dict["def_action"]]
    # print(f"def_action {global_def_action}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/def_action_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_def_action, the_file)
    # the_file.close()
