import torch
import torch.multiprocessing as mp
import torch.nn as nn
#
# class Agent(mp.Process):
#     def __init__(self):
#         super(Agent, self).__init__()
#         self.model = torch.nn.Sequential(torch.nn.Linear(3, 2))
#
#     def run(self):
#         temp_tensor = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
#         print(self.model(temp_tensor))
#
# num_worker = 1
# workers = [Agent() for i in range(num_worker)]
#
# [w.start() for w in workers]
# [w.join() for w in workers]


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 2))
        print("init")
        # self.run()

    def run(self):
        temp_tensor = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        print("here")
        print(self.net(temp_tensor))

    # def forward(self):
    #     out = self.net(torch.tensor([[1., 2., 3.], [1., 2., 3.]]))
    #     return out

def help_func():
    model = Agent()
    model.run()


if __name__ == '__main__':
    num_processes = 4
    # NOTE: this is required for the ``fork`` method to work
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=help_func, args=())
        p.start()
        processes.append(p)
    for p in processes:
        p.join()



