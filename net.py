from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils import *


class InferenceNetwork(nn.Module):

    def __init__(self, hw_embed_on, hw_embed_dim, layer_size, determ):
        super(InferenceNetwork, self).__init__()
        #self.z_on = args.z_on
        self.num_channel = 3
        #self.with_sampling = None
        self.hw_embed_on = hw_embed_on
        self.layer_size = layer_size
        self.determ = determ

        # z encoder
        self.z_encoder = nn.Sequential(*[
            nn.Linear(hw_embed_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 2*(self.layer_size*2+1)*3)
        ])

        self.softplus = nn.Softplus().cuda()
        self.softmax = nn.Softmax(dim=0).cuda()


    def get_posterior(self, inputs):
        (x, y, hw_embed) = inputs
        z_stats = self.z_encoder(hw_embed)
        mu_z = z_stats[:(self.layer_size*2+1)*3].squeeze()     # even indices for mean
        sigma_z = z_stats[(self.layer_size*2+1)*3:].squeeze()  # odd indices for sigma
        q_z = torch.distributions.Normal(mu_z, self.softplus(sigma_z))

        return q_z

    def forward(self, inputs):
        # compute posterior
        q_z = self.get_posterior(inputs)
        # compute kl
        kl_z     = torch.sum(kl_diagnormal_stdnormal(q_z))
        # sample variables from the posterior
        z = None
        kl = 0.

        kl = kl + kl_z 
        z_ = q_z.rsample() if not self.determ else q_z.mean 
        zw_ = z_[:(self.layer_size*2)*3].squeeze()     # even indices for weights 
        zb_ = z_[(self.layer_size*2)*3:].squeeze()     # odd indices for biases 
        z = {'w':zw_, 'b':zb_}

        return z, kl 

def kl_diagnormal_stdnormal(p):
    pshape = p.mean.shape
    device = p.mean.device
    q = torch.distributions.Normal(torch.zeros(pshape, device=device), torch.ones(pshape, device=device))
    return torch.distributions.kl.kl_divergence(p, q).to(device)


class MetaLearner(nn.Module):

    def __init__(self, hw_embed_on, 
                    hw_embed_dim, 
                    layer_size):
        super(MetaLearner, self).__init__()

       
        self.meta_learner = Net(nfeat=132, 
                                    hw_embed_on=hw_embed_on,
                                    hw_embed_dim=hw_embed_dim, 
                                    layer_size=layer_size)


    def forward(self, X, hw_embed, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X, hw_embed)
        else:
            out = self.meta_learner(X, hw_embed, adapted_params)
        return out

    def cloned_params(self):
        params = OrderedDict()
        for (key, val) in self.named_parameters():
            params[key] = val.clone()
        return params


class Net(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, nfeat, hw_embed_on, hw_embed_dim, layer_size):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.hw_embed_on = hw_embed_on

        self.add_module('fc1', nn.Linear(nfeat, layer_size))
        self.add_module('fc2', nn.Linear(layer_size, layer_size))

        if hw_embed_on:
            self.add_module('fc_hw1', nn.Linear(hw_embed_dim, layer_size))
            self.add_module('fc_hw2', nn.Linear(layer_size, layer_size))
            hfeat = layer_size * 2 
        else:
            hfeat = layer_size

        self.add_module('fc3', nn.Linear(hfeat, hfeat))
        self.add_module('fc4', nn.Linear(hfeat, hfeat))

        self.add_module('fc5', nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hw_embed=None, params=None):
        hw_embed = hw_embed.repeat(len(x), 1)
        if params == None:
            out = self.relu(self.fc1(x))
            out = self.relu(self.fc2(out))

            if self.hw_embed_on:
                hw = self.relu(self.fc_hw1(hw_embed))
                hw = self.relu(self.fc_hw2(hw))
                out = torch.cat([out, hw], dim=-1)

            out = self.relu(self.fc3(out))
            out = self.relu(self.fc4(out))
            out = self.fc5(out)

        else:
            out = F.relu(F.linear(x, params['meta_learner.fc1.weight'],
                                params['meta_learner.fc1.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc2.weight'],
                                params['meta_learner.fc2.bias']))
            
            if self.hw_embed_on:
                hw = F.relu(F.linear(hw_embed, params['meta_learner.fc_hw1.weight'],
                                    params['meta_learner.fc_hw1.bias']))
                hw = F.relu(F.linear(hw, params['meta_learner.fc_hw2.weight'],
                                    params['meta_learner.fc_hw2.bias']))
                out = torch.cat([out, hw], dim=-1)

            out = F.relu(F.linear(out, params['meta_learner.fc3.weight'],
                                params['meta_learner.fc3.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc4.weight'],
                                params['meta_learner.fc4.bias']))
            out = F.linear(out, params['meta_learner.fc5.weight'],
                                params['meta_learner.fc5.bias']) 

        return out
