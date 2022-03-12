# OS library
import os
# Pytorch 
import torch

from scipy.stats import spearmanr, kendalltau, pearsonr

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr

import logging 

import wandb

def set_seed(args):
    """
    Set the seed for result reproducibility 

    Input 
    args : User Arguments 
    """

    # Pytorch seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_gpu(args):
    """
    Set number of visible GPU 
    
    Input 
    args : User specified arguments
    Return 
    args 
    """
    os.environ['CUDA_VISIBLE_DEVICES']= '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args 


def set_path(args):
    """
    Set the datapaths 
    Input 
    args : User specified arguments
    Return 
    args  
    """
    args.data_path = os.path.join(args.main_path , 'data')
    args.save_path = os.path.join(args.save_path , 'train_result')
    args.save_path = os.path.join(args.save_path , args.exp_name)

    # If save file path doesnt exists create file path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'checkpoint'))
    print(f"==> save path is [{args.save_path}] ...")
    return args




class Log():
    def __init__(self, save_path, summary_steps, metrics, devices, split, writer=None, use_wandb=False):
        self.save_path = save_path
        self.metrics = metrics
        self.devices = devices
        self.summary_steps = summary_steps
        self.split = split
        self.writer = writer

        self.epi = []
        self.elems = {}
        for metric in metrics:  
            self.elems[metric] = { device: [] for device in devices }
        self.elems['loss'] = { device: [] for device in devices }
        self.elems['mse_loss'] = { device: [] for device in devices }
        self.elems['kl_loss'] = { device: [] for device in devices }
        # self.elems['denorm_mse'] = { device: [] for device in devices }

        self.use_wandb = use_wandb

    def update_epi(self, i_epi):
        self.epi.append(i_epi)

    def update(self, i_epi, metric, device, val):
        self.elems[metric][device].append(val)
        if self.use_wandb:
            log_dict = {f'{self.split}_{metric}/{device}': val}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None:
            self.writer.add_scalar(f'{self.split}_{metric}/{device}', val, i_epi)  

    def avg(self, i_epi, metric, is_print=True):
        v = 0.0
        cnt = 0
        for device in self.devices:
            v += self.get(metric, device, i_epi)
            cnt += 1     
        if self.use_wandb:
            log_dict = {f'mean/{self.split}_{metric}': v / cnt}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None and is_print:
            self.writer.add_scalar(f'mean/{self.split}_{metric}', v / cnt, i_epi)
        return v / cnt
    


    def get(self, metric, device, i_epi):
        idx = self.epi.index(i_epi)
        return self.elems[metric][device][idx]

    def save(self):
        torch.save({
                    'summary_steps': self.summary_steps,
                    'episode': self.epi,
                    'elems': self.elems
                    }, 
                    os.path.join(self.save_path, f'{self.split}_log_data.pt'))


metrics_fn = {
            'spearman': lambda yq_hat, yq: spearmanr(flat(yq_hat), flat(yq)),
            'pearsonr': lambda yq_hat, yq: pearsonr(flat(yq_hat), flat(yq)),
            'kendalltau': lambda yq_hat, yq: kendalltau(flat(yq_hat), flat(yq))
            }

def flat(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().reshape(-1)
    else:
        return v.reshape(-1)

def normalization(latency , index= None , portion = 0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])

    else:
        min_val = min(latency)
        max_val = max(latency)

    latency = (latency - min_val) / (max_val - min_val) * portion + (1-portion) /2
    return latency







# Defining the loss functions as the lambda functions
loss_fn = {
            'mse': lambda yq_hat, yq,: F.mse_loss(yq_hat, yq),
            }


def arch_enc(arch):
    feature=[]
    for i in arch:
        onehot = np.zeros(6)
        if i == 8 :
            feature = np.hstack([feature, onehot])
        else :
            if i < 4:
                onehot[0] = 1
            elif i < 8:
                onehot[1] = 1
            k = i % 4
            onehot[2+k] = 1
            feature = np.hstack([feature, onehot])
    assert len(feature) == 132
    return torch.FloatTensor(feature)

def set_logger(log_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)