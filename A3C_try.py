# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C
import os
import pickle
import time
from copy import copy

import gym
import torch as torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Gym_HoneyDrone import HyperGameSim
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi2 = nn.Linear(128, 256)
        self.v2 = nn.Linear(128, 256)
        self.pi3 = nn.Linear(256, 128)
        self.v3 = nn.Linear(256, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi2 = self.pi2(pi1)
        v2 = self.v2(v1)

        pi3 = F.relu(self.pi3(pi2))
        v3 = F.relu(self.v3(v2))


        pi = self.pi(pi3)
        v = self.v(v3)

        return pi, v

    def calc_R(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_id, glob_episode_thred, global_dict):
        super(Agent, self).__init__()
        # self.env = gym.make(env_id)
        self.env = HyperGameSim()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Local Using", device)
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma).to(self.device)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        print("creating: "+self.name)
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.glob_episode_thred = glob_episode_thred
        self.shared_dict = global_dict
        # self.writer = writer

    def run(self):
        # create writer for TensorBoard
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        writer = SummaryWriter("runs/"+self.shared_dict["start_time"])

        while self.episode_idx.value < self.glob_episode_thred:
            # Episode start
            t_step = 1
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                self.shared_dict["def_action"][action] += 1
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % 5 == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1

            # this one commented out for save memory, uncomment as needed.
            # save data
            # self.shared_dict["reward"].append(score)
            # self.shared_dict["t_step"].append(t_step)       # number of round for a game


            print(self.name, 'global-episode ', self.episode_idx.value, 'reward %.1f' % score)
            writer.add_scalar("Score", score, self.episode_idx.value)

        writer.flush()
        writer.close()  # close SummaryWriter of TensorBoard
        return



if __name__ == '__main__':

    start_time = time.time()
    lr = 1e-4
    env_id = 'CartPole-v0'
    # temp_env = gym.make(env_id)
    temp_env = HyperGameSim()
    n_actions = temp_env.action_space.n
    input_dims = temp_env.observation_space.shape
    T_MAX = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)
    global_actor_critic = ActorCritic(input_dims, n_actions).to(device)        # global NN
    print(global_actor_critic)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                       betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    glob_episode_thred = 1  # total number of episodes to run
    num_worker = 2 # mp.cpu_count()     # update this for matching computer resources

    shared_dict = {}
    shared_dict["reward"] = Manager().list()  # use Manager().list() to create a shared list between processes
    shared_dict["t_step"] = Manager().list()
    shared_dict["def_action"] = mp.Array('i', n_actions)
    shared_dict["start_time"] = str(start_time)
    # shared_dict["reward"] = mp.Array('d', glob_episode_thred + num_worker)        # save simulation data ('d' means double-type)



    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id, glob_episode_thred=glob_episode_thred, global_dict=shared_dict) for i in range(num_worker)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    # Saving data to file
    # Reward
    global_reward = [ele for ele in shared_dict["reward"]]     # convert mp.Array to python list
    print(f"reward {global_reward}")
    os.makedirs("data/A3C", exist_ok=True)
    the_file = open("data/A3C/reward_train_all_result.pkl",
                    "wb+")
    pickle.dump(global_reward, the_file)
    the_file.close()

    # t_step (number of round in a game)
    global_t_step = [ele for ele in shared_dict["t_step"]]
    print(f"t_step {global_t_step}")
    os.makedirs("data/A3C", exist_ok=True)
    the_file = open("data/A3C/t_step_all_result.pkl",
                    "wb+")
    pickle.dump(global_t_step, the_file)
    the_file.close()

    # defender action frequency
    global_def_action = [ele for ele in shared_dict["def_action"]]
    print(f"def_action {global_def_action}")
    os.makedirs("data/A3C", exist_ok=True)
    the_file = open("data/A3C/def_action_all_result.pkl",
                    "wb+")
    pickle.dump(global_def_action, the_file)
    the_file.close()

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))


