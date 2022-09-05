'''
Project     ：gym-drones 
File        ：A3C_model_3.py
Author      ：Zelin Wan
Date        ：9/3/22
Description : 
'''

'''
Project     ：gym-drones 
File        ：A3C_try_2.py
Author      ：Zelin Wan
Date        ：9/1/22
Description : 
'''

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim, fixed_seed=True):
        super(ActorCritic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)

        set_init([self.pi1, self.pi2, self.v1, self.v2], fixed_seed)
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        # print("v_t", v_t)
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        # print(c_loss, a_loss, total_loss)
        return total_loss, c_loss.mean(), a_loss.mean()


class Agent(mp.Process):
    def __init__(self, gnet, opt, shared_dict, gamma, MAX_EP, fixed_seed, trial, name_id: int, player='def'):
        super(Agent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        self.gnet = gnet
        self.opt = opt
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        self.g_r_list = shared_dict['glob_r_list']
        self.env = gym.make('CartPole-v1')
        self.N_S = self.env.observation_space.shape[0]
        self.N_A = self.env.action_space.n
        self.lnet = ActorCritic(self.N_S, self.N_A, fixed_seed)  # local network
        self.gamma = gamma
        self.MAX_EP = MAX_EP
        self.trial = trial
        self.player = player


    def run(self):
        # ======== Create Writer for TensorBoard ========
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        if self.name_id == 0:
            if self.trial is not None:
                trial_num_str = str(self.trial.number)
            else:
                trial_num_str = "None"

            if self.shared_dict["on_server"]:
                path = "/home/zelin/Drone/code_files/data/"
            else:
                path = ""
            writer = SummaryWriter(log_dir=path + "runs_" + self.player + "/each_run_" + self.shared_dict["start_time"] + "-" +
                                   self.player + "-" + "-Trial_" + trial_num_str + "-eps")
            # writer = None
            # print("creating writer", "runs_"+self.player+"/each_run_" + self.shared_dict["start_time"] + "-" + self.player + "-" + "-Trial_" + trial_num_str + "-eps")

        else:
            writer = None


        total_step = 1
        while self.g_ep.value < self.MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            total_loss_set = []
            c_loss_set = []
            a_loss_set = []
            score = 0
            oppo_score = 0
            done = False
            while not done:
                # if self.name == 'w00':
                    # self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                score += r
                oppo_score += 0     # TODO: change this when use Drone's Env
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % 5 == 0 or done:  # update global and assign to local net
                    # sync
                    total_loss, c_loss, a_loss = push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s,
                                                               buffer_a, buffer_r, self.gamma)
                    total_loss_set.append(total_loss.item())
                    c_loss_set.append(c_loss.item())
                    a_loss_set.append(a_loss.item())
                    buffer_s, buffer_a, buffer_r = [], [], []

                s = s_
                total_step += 1

            # save data to shared dictionary for tensorboard
            with self.g_ep.get_lock():
                self.shared_dict['eps_writer'].put(self.g_ep.value)
                self.g_ep.value += 1
            self.g_r_list.append(score)
            self.shared_dict["score_writer"].put(score)
            self.shared_dict["oppo_score_writer"].put(oppo_score)
            self.shared_dict["lr_writer"].put(self.opt.param_groups[0]['lr'])
            self.shared_dict['t_loss_writer'].put(sum(total_loss_set)/len(total_loss_set))
            self.shared_dict['c_loss_writer'].put(sum(c_loss_set) / len(c_loss_set))
            self.shared_dict['a_loss_writer'].put(sum(a_loss_set) / len(a_loss_set))


            # ==== tensorboard writer ====
            # Only agent (index 0) can write to tensorboard

            if writer is not None:
                while not self.shared_dict['eps_writer'].empty():
                    # use episode as index of tensorboard
                    current_eps = self.shared_dict['eps_writer'].get()
                    # write score
                    writer.add_scalar("Score", self.shared_dict["score_writer"].get(), current_eps)
                    writer.add_scalar("Opponent's Score", self.shared_dict["oppo_score_writer"].get(), current_eps)
                    # write lr
                    writer.add_scalar("Learning rate", self.shared_dict["lr_writer"].get(), current_eps)
                    # write epsilon
                    # writer.add_scalar("Epsilon (random action probability)", self.shared_dict["epsilon"][id],
                    #                   current_eps)
                    # # write mission time (step)
                    # writer.add_scalar("Mission Time (step)", t_step, self.shared_dict["eps"][id])
                    # write loss
                    writer.add_scalar("Model's Total Loss", self.shared_dict['t_loss_writer'].get(), current_eps)
                    writer.add_scalar("Model's Critic Loss", self.shared_dict['c_loss_writer'].get(), current_eps)
                    writer.add_scalar("Model's Actor Loss", self.shared_dict['a_loss_writer'].get(), current_eps)
                    # mission completion rate
                    # writer.add_scalar("Mission Success Rate (completion rate)", self.env.system.scanCompletePercent(), self.shared_dict["eps"][id])   # TODO: comment out for test




                # for id in range(self.shared_dict["index"], len(self.shared_dict["eps"])):
                #     # write score
                #     writer.add_scalar("Average Score", self.shared_dict["score"][id], self.shared_dict["eps"][id])
                #     writer.add_scalar("Opponent's Average Score", self.shared_dict["oppo_score"][id],
                #                       self.shared_dict["eps"][id])
                #     # write lr
                #     writer.add_scalar("Learning rate", self.shared_dict["lr"][id], self.shared_dict["eps"][id])
                    # # write epsilon
                    # writer.add_scalar("Epsilon (random action probability)", self.shared_dict["epsilon"][id],
                    #                   self.shared_dict["eps"][id])
                    # # write mission time (step)
                    # writer.add_scalar("Mission Time (step)", t_step, self.shared_dict["eps"][id])
                    # # write loss
                    # writer.add_scalar("Model's Total Loss", sum(total_loss_set) / len(total_loss_set),
                    #                   self.shared_dict["eps"][id])
                    # writer.add_scalar("Model's Critic Loss", sum(critic_loss_set) / len(critic_loss_set),
                    #                   self.shared_dict["eps"][id])
                    # writer.add_scalar("Model's Actor Loss", sum(actor_loss_set) / len(actor_loss_set),
                    #                   self.shared_dict["eps"][id])
                    # mission completion rate
                    # writer.add_scalar("Mission Success Rate (completion rate)", self.env.system.scanCompletePercent(), self.shared_dict["eps"][id])   # TODO: comment out for test
                # update index
                # self.shared_dict["index"] = len(self.shared_dict["eps"])

            # if self.name_id == 0:
            #     self.scheduler.step()  # update learning rate each episode
            print(self.name, 'episode ', self.g_ep.value, 'reward %.1f' % score)
        # self.res_queue.put(None)

        if writer is not None:
            writer.flush()
            writer.close()  # close SummaryWriter of TensorBoard
        return