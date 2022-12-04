from collections import defaultdict
from dataclasses import replace
from email.policy import strict
import math
import random

import utility
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm
import wandb
import collections
import copy
class Trainer():
    def __init__(self, args, loader, agent_list, ckp):
        self.args = args
        self.ckp = ckp #save need modified
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loaders_train = loader.loaders_train
        self.agent_list = agent_list[:-1] #last one is the tester
        self.tester = agent_list[-1]
        self.epoch = 0
        for agent in self.agent_list:
            agent.make_optimizer_all(ckp=ckp)
            agent.make_scheduler_all()
        self.tester.make_optimizer_all(ckp=ckp)
        self.tester.make_scheduler_all()
        
        self.run=wandb.init(project=args.project)
        self.run.name=args.save

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        self.sync_at_init()

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        # epoch, _ = self.start_epoch()
        self.epoch += 1
        epoch = self.epoch
        
        for agent in self.agent_list:
            agent.make_optimizer_all()
            agent.make_scheduler_all(reschedule=epoch-1) #pls make sure if need resume  
        # Step 1: sample a list of agent w/o replacement
        agent_joined = np.sort(np.random.choice(range(self.args.n_agents),self.args.n_joined, replace=False))
        # Step 2: sample a list of associated budgets
        while True:
            agent_budget = np.random.choice(self.args.fraction_list, self.args.n_joined)
            # For implementation simiplicity, we sample all model size for all methods
            _, unique_counts = np.unique(agent_budget, return_counts=True)
            if len(unique_counts) == len(self.args.fraction_list):
                break
        # Step 3: create a buget -> client dictionary for syncing later
        budget_record = collections.defaultdict(list) #need to make sure is not empty
        for k, v in zip(agent_budget, agent_joined):
            budget_record[k].append(v)
        
        for i in agent_joined:
            self.agent_list[i].begin_all(epoch, self.ckp) #call move all to train() 
            self.agent_list[i].start_loss_log() #need check!

        self.start_epoch() #get current lr
        timer_model = utility.timer()

        for idx, i in enumerate(agent_joined):
            timer_model.tic()

            loss, loss_orth, log_train = self.agent_list[i].train_local(self.loaders_train[i], 
                              agent_budget[idx], self.args.local_epochs)
            
            timer_model.hold()
            tt=timer_model.release()
            
            self.ckp.write_log(
                    '{}/{} ({:.0f}%)\t'
                    'agent {}\t'
                    'model {}\t'
                    'NLL: {:.3f}\t'
                    'Top1: {:.2f} / Top5: {:.2f}\t'
                    'Total {:<2.4f}/ Orth: {:<2.5f} '
                    'Time: {:.1f}s'.format(
                        idx+1,
                        len(agent_joined),
                        100.0 * (idx+1) / len(agent_joined),i,agent_budget[idx],
                        *(log_train),
                        loss, loss_orth,
                        tt
                        )
                    )
                  
        for i in agent_joined:
            self.agent_list[i].log_all(self.ckp)
            for loss in self.agent_list[i].loss_list:
                loss.end_log(len(self.loader_train.dataset)*self.args.local_epochs) #should be accurate
        
        
        self.sync(budget_record, agent_joined, agent_budget)


    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        self.tester.test_all(self.loader_test, timer_test, self.run, epoch)


    def sync_at_init(self): 
        if self.args.resume_from:
            for i in range(len(self.args.fraction_list)):
                print("resume from checkpoint")
                self.tester.model_list[i].load_state_dict(torch.load('../experiment/'+self.args.save+'/model/model_m'+str(i)+'_'+self.args.resume_from+'.pt'))
        # Sync all agents' parameters with tester before training 
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            state_dict = self.tester.model_list[model_id].state_dict()
            for agent in self.agent_list:
                agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict),strict=True)
            
    def sync(self, budget_record, agent_joined, agent_budget):
        # Step 1: gather all filter banks
        # This step runs across network fractions
        filter_banks = {}
        for k,v in self.tester.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape)
        
        for k in filter_banks:
            for b, i in zip(agent_budget,agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                state_dict = self.agent_list[i].model_list[model_id].state_dict()
                filter_banks[k]+=state_dict[k]*(1./self.args.n_joined)
        # Step 2: gather all other parameters
        # This step runs within each network fraction
        anchors = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]
            
            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)
            for k in anchor:
                for i in agent_list_at_net_f:
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    state_dict = self.agent_list[i].model_list[model_id].state_dict()
                    anchor[k]+=state_dict[k]*(1./n_models)
            anchors[net_f] = anchor 

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        
        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        return filter_banks, anchors

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):

        lr = self.agent_list[0].scheduler_list[0].get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(self.epoch, lr))

        return self.epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

