import torch
import os
import numpy as np
from utils import *



class Data:
    def __init__(self , data_path , meta_train_devices,meta_valid_devices,
                    num_inner_tasks, 
                    num_meta_train_sample,
                    num_sample, 
                    num_query,
                    num_query_meta_train_task=200,
                    remove_outlier=True):
    
        # List of hardware training devices parsed from argument parser
        self.meta_train_devices = meta_train_devices
        # Datapath
        self.datapath = data_path
        #List of hardware validation devices parsed from argument parser
        self.meta_valid_devices = meta_valid_devices
        # number of inner tasks for a episode
        self.num_inner_tasks = num_inner_tasks
        # Number of training sample for each device in task pool
        self.num_meta_train_sample = num_meta_train_sample
        # Number of trainign sample for each task - in finetuning stage 
        self.num_sample = num_sample
        # Number of test(query)
        self.num_query = num_query
        # Remove outlier values
        self.remove_outlier = remove_outlier
        # Number of architectures to train on
        nts = self.num_meta_train_sample
        self.num_query_meta_train_task = num_query_meta_train_task

        # Loading all the architectures (Getting their embedding)
        self.load_archs()

        # Dictionary containing devices and their latency on specific architectures in the search space
        self.latency = {}
        

        # Dictionary of devices and their corresponding architectures we need to train on
        self.train_idx = {}
        self.valid_idx = {}

        self.norm_latency = {}
        # Important loop loading latency values of the device
        for device in self.meta_train_devices + self.meta_valid_devices:
            # Latency of Device is the Latency values of the device calculated on some architectures in the search space there are currently 5000 architectures so 5000 latencies
            self.latency[device] = torch.FloatTensor(torch.load(os.path.join(self.datapath , 'latency' , f'{device}.pt')))
            # Some reference architectures for each device used for training out of 5000 architectures with only their indices 
            self.train_idx[device] = torch.arange(len(self.archs))[:nts]
            self.valid_idx[device] = torch.arange(len(self.archs))[nts : nts + num_query]
            
            # Normalizing the latency values according to training architectures
            self.norm_latency[device] = normalization(self.latency[device] , index = self.train_idx[device])

        # Load index set of reference architectures
        self.hw_embed_idx = torch.load(os.path.join(self.datapath , 'hardware_embedding_index.pt'))
    print("==> load data ...")



    def load_archs(self):
        self.archs = [arch_enc(_['op_idx_list']) for _ in torch.load(os.path.join(self.datapath , 'metainfo.pt'))['arch']]


    def generate_episode(self):
        # Episode list
        episode = []
        # Randomly selecting some devices as the meta batch for training network
        #num_inner_task denotes number of tasks in a episode 
        # Say 10 meta train devices so selecting 8 devices for a batch 
        rand_device_idx = torch.randperm(len(self.meta_train_devices))[:self.num_inner_tasks]
        for t in rand_device_idx:
            # sample device randomly selected for a task 
            device = self.meta_train_devices[t]
            # hardware embedding of the device 
            latency = self.latency[device]
            # Calculating the hardware embedding that is unormalized
            hw_embed = latency[self.hw_embed_idx]
            #normalizing the hardware embedding 
            hw_embed = normalization(hw_embed , portion = 1.0)

            # Sample for finetuning and test(query)
            rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
            finetune_idx = rand_idx[:self.num_sample]
            qry_idx = rand_idx[self.num_sample : self.num_sample + self.num_query_meta_train_task]


            # Getting x_finetune and x_query
            x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
            x_query = torch.stack([self.archs[_] for _ in qry_idx])

            # Calculating the latency values
            y_finetune = self.norm_latency[device][finetune_idx].view(-1 , 1)
            y_query = self.norm_latency[device][qry_idx].view(-1 , 1)

            episode.append((hw_embed , x_finetune , y_finetune , x_query , y_query , device))
        return episode

    def generate_test_tasks(self, split=None):
        if split == 'meta_train':
            device_list = self.meta_train_devices
        elif split == 'meta_valid':
            device_list = self.meta_valid_devices
        elif split == 'meta_test':
            device_list = self.meta_test_devices
        else: NotImplementedError

        tasks = []
        for device in device_list:
            tasks.append(self.get_task(device))
        return tasks
        
    def get_task(self, device=None, num_sample=None):
        if num_sample == None:
            num_sample = self.num_sample
    
        latency = self.latency[device]
        # hardware embedding
        hw_embed = latency[self.hw_embed_idx]
        hw_embed = normalization(hw_embed, portion=1.0)        
        
        # samples for finetuing & test (query)
        rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
        finetune_idx = rand_idx[:num_sample]

        x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
        x_qry = torch.stack([self.archs[_] for _ in self.valid_idx[device]])
       
        y_finetune = self.norm_latency[device][finetune_idx].view(-1, 1)
        y_qry = self.norm_latency[device][self.valid_idx[device]].view(-1, 1)

        return hw_embed, x_finetune, y_finetune, x_qry, y_qry, device

