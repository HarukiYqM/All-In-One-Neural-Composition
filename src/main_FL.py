import torch

import utility
from data import Data
from model import Model
from loss import Loss
from trainer_FL import Trainer
from option import args
import random
import numpy as np
from model.agent_federation import Agent


random.seed(0)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
np.random.seed(0)
#print('Flag')
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministics=True

# This file create a list of agents...
# It also create a tester that sync at each communication for benchmarking
if checkpoint.ok:
    loader = Data(args)
    agent_list = [Agent(args, checkpoint, my_id) for my_id in range(args.n_agents)] #share ckp...need check if save
    tester = Agent(args, checkpoint, 1828) #a tester for runing test. Assign it a fixed id
    for agent in agent_list:
        agent.make_loss_all(Loss)
    tester.make_loss_all(Loss)
    #loss = Loss(args, checkpoint)
    t = Trainer(args, loader, agent_list+[tester], checkpoint)
    while not t.terminate():
        if agent_list[0].scheduler_list[0].last_epoch == -1 and not args.test_only:
            t.test()
        t.train()
        t.test()

    checkpoint.done()
