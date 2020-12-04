import torch.nn as nn
import torch
import torch.nn.functional as F
from    torch import optim
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np
import  torch.autograd as autograd
from    copy import deepcopy
import conv4


class MetaLearner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.encoder = conv4.Conv4()
        z_dim = 1600
        self.base_net = conv4.Linear_fw(z_dim, self.args.way)
        self.pre_num = self.args.shot * self.args.way

    def forward(self, data_list, label_shot_list):
        out_list = list( map(self.single_task_out, zip(data_list, label_shot_list) )  )
        out_list = torch.stack(out_list, dim = 0)
        return out_list

    def set_fast(self, x):
        x.fast = x

    def all_set_fast(self, x_list):
        list(map(self.set_fast, x_list))

    def update_fast(self, x_list):
        weight, grad = x_list
        weight.fast = weight.fast - grad * self.update_lr

    def single_task_out(self, single_task_data):
        data_in, label_shot = single_task_data
        data_shot = data_in[:self.pre_num]
        data_query = data_in[self.pre_num:]
        
        tot_parameters = list(self.encoder.parameters()) + list(self.base_net.parameters())
        self.all_set_fast(tot_parameters)

        for _ in range(self.update_step):
            feature_shot = self.encoder(data_shot)
            out_shot = self.base_net(feature_shot)
            loss = F.cross_entropy(out_shot, label_shot)
            grads = autograd.grad(loss, list(map(lambda p: p.fast, tot_parameters)), create_graph = False)
            list(map(self.update_fast, zip(tot_parameters, grads)))
        pass
        feature_query = self.encoder(data_query)
        out_query = self.base_net(feature_query)
        return out_query