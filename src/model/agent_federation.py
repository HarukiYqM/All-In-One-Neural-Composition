import enum
import fractions
import os
from importlib import import_module
from sched import scheduler
import torch
import torch.nn as nn
from IPython import embed
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from tqdm import tqdm
import copy
class Agent:
    def __init__(self, *args):
        super(Agent, self).__init__()
        print('Init Agent {} and making models...'.format(args[2]))

        self.args = args[0] # args should contain slim rate
        self.ckp = args[1]
        self.my_id = args[2]
        self.crop = self.args.crop
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.precision = self.args.precision
        self.n_GPUs = self.args.n_GPUs
        self.save_models = self.args.save_models
        self.fractions = self.args.fraction_list

        # To do... If not use resume, share ckp may be safe
        

        print("Init a List of Models")
        model_list = []
        self.budget_model = {net_f:i for i,net_f in enumerate(self.fractions)}
        
        for net_f in self.fractions:
            module = import_module('model.' + self.args.model.lower())
            self.module = module
            new_args = self.args
            new_args.net_fraction = net_f
            model_list.append(module.make_model(new_args))
        self.model_list = model_list
       
        print("Filter bank synced at initalization!")
        self.sync_at_init()
        if not self.args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
        
        # temporarily disable data parallel

        self.load_all(
            self.ckp.dir,
            pretrained=self.args.pretrained,
            load=self.args.load,
            resume=self.args.resume,
            cpu=self.args.cpu
        )
        
        for i, m in enumerate(self.model_list):
            print(self.get_model(i), file=self.ckp.log_file)
        
        self.summarize(self.ckp)

    def test_all(self, loader_test, timer_test, run, epoch):
        timer_test = timer_test

        for i, model in enumerate(self.model_list):
            self.model_list[i] = self.model_list[i].to(self.device)
            self.loss_list[i].start_log(train=False)
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(loader_test, ncols=80):

                    img, label = self.prepare(img, label)
                    torch.cuda.synchronize()
                    timer_test.tic()

                    prediction = model(img)
                    torch.cuda.synchronize()
                    timer_test.hold()

                    self.loss_list[i](prediction, label, train=False)

            self.loss_list[i].end_log(len(loader_test.dataset), train=False)
            best = self.loss_list[i].log_test.min(0)
            self.model_list[i] = self.model_list[i].to('cpu')
            for j, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log(
                    'model {} {}: {:.3f} (Best: {:.3f} from epoch {})'.format(
                        i,
                        measure,
                        self.loss_list[i].log_test[-1, j],
                        best[0][j],
                        best[1][j] + 1 if len(self.loss_list[i].log_test) == len(self.loss_list[i].log_train) else best[1][j]
                        )
                    )
            run.log({"acc @ {}".format(self.fractions[i]): 100-self.loss_list[i].log_test[-1, self.args.top]},step=epoch-1)
            total_time = timer_test.release()
            is_best = self.loss_list[i].log_test[-1, self.args.top] <= best[0][self.args.top]
            self.ckp.save(self, i, epoch, is_best=is_best)
            #self.ckp.save_results(epoch, i, model)
            self.scheduler_list[i].step()

    def budget_to_model(self, budget):
        return self.budget_model[budget]
    def train_local(self, loader_train, budget, epochs):
        
        model_id = self.budget_to_model(budget)
        loss_list = []
        loss_orth_list = []
        n_samples = 0
        self.model_list[model_id] = self.model_list[model_id].to(self.device)
        for epoch in range(epochs):
            for batch, (img, label) in enumerate(loader_train):                
                img, label = self.prepare(img, label)
                n_samples += img.size(0)

                self.optimizer_list[model_id].zero_grad()
                prediction = self.forward(img, model_id)
                loss, _ = self.loss_list[model_id](prediction, label,)
                    
                
                loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[model_id],self.args,'L2')
                loss = loss_orth + loss
                loss_orth_list.append(loss_orth.item())
                                    
                loss.backward()
                self.optimizer_list[model_id].step()
                
                loss_list.append(loss.item())

        log_train = self.loss_list[model_id].log_train[-1,:]/n_samples
        self.model_list[model_id] = self.model_list[model_id].to('cpu')
        
        return sum(loss_list)/len(loss_list), sum(loss_orth_list)/len(loss_orth_list), log_train

    
    def train_one_step(self, img, label):
        loss_list = []
        loss_orth_list = []
        for i, _ in enumerate(self.model_list):
            self.optimizer_list[i].zero_grad()
            prediction = self.forward(img, i)
            loss, _ = self.loss_list[i](prediction, label,)
            
            
            loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[i],self.args,'L2')
            loss = loss_orth + loss
            loss_orth_list.append(loss_orth.item())
                            
            loss.backward()
            self.optimizer_list[i].step()
            
            loss_list.append(loss.item())
        if self.args.sync: self.sync_filter()
        
        return loss_list, loss_orth_list
    
    def sync_at_init(self):
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = v
        
        for i in range(n_models):
            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)
    

    def sync_filter(self):
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape).cuda()
        
        for k in filter_banks:
            for model in self.model_list:  
                state_dict = model.state_dict()
                filter_banks[k]+=state_dict[k]*(1./n_models)
        
        for i in range(n_models):
            
            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)

    def forward(self, x, i):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        # from IPython import embed; embed(); exit()
        x = self.model_list[i](x)

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x
    
    def get_model(self, i):
        if self.n_GPUs == 1:
            return self.model_list[i]
        else:
            return self.model_list[i].module

    def state_dict_all(self, **kwargs):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.state_dict(i))
        return ret
    
    def state_dict(self, i, **kwargs):
        return self.get_model(i).state_dict(**kwargs)

    def save_all(self, apath, epoch, is_best=False):
        for i, _ in enumerate(self.model_list):
            self.save(i, apath, epoch, is_best)

    def save(self, i, apath, epoch, is_best=False):
        target = self.get_model(i).state_dict()

        conditions = (True, is_best, self.save_models)
        names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,n))
                )
    
    def load_all(self, apath, pretrained='', load='', resume=-1, cpu=False):
        for i, _ in enumerate(self.model_list):
            self.load(i, apath, pretrained, load, resume, cpu)
    
    def load(self, i, apath, pretrained='', load='', resume=-1, cpu=False):
        f = None
        if pretrained:
            if pretrained != 'download':
                print('Load pre-trained model from {}'.format(pretrained))
                f = pretrained
                # from IPython import embed; embed(); exit()
        else:
            if load:
                if resume == -1:
                    print('Load model {} after the last epoch'.format(i))
                    resume = 'latest'
                else:
                    print('Load model {} after epoch {}'.format(i,resume))

                f = os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,resume))

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}

            state = torch.load(f, **kwargs)
            # from IPython import embed; embed(); exit()

            self.get_model(i).load_state_dict(state, strict=False)

    def begin_all(self, epoch, ckp):
        for i, _ in enumerate(self.model_list):
            self.begin(i, epoch, ckp)

    def begin(self, i, epoch, ckp):
        self.model_list[i].train()
        m = self.get_model(i)
        if hasattr(m, 'begin'): m.begin(epoch, ckp)

    def start_loss_log(self):
        for loss in self.loss_list:
            loss.start_log() #create a tensor
    
    def log_all(self, ckp):
        for i, _ in enumerate(self.model_list):
            self.log(i, ckp)
    
    def log(self, i, ckp):
        m = self.get_model(i)
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        for i, _ in enumerate(self.model_list):
            ckp.write_log('# parameters of model {}: {:,}'.format(i,
                sum([p.nelement() for p in self.model_list[i].parameters()])
            ))

            kernels_1x1 = 0
            kernels_3x3 = 0
            kernels_others = 0
            gen = (c for c in self.model_list[i].modules() if isinstance(c, nn.Conv2d))
            for m in gen:
                kh, kw = m.kernel_size
                n_kernels = m.in_channels * m.out_channels
                if kh == 1 and kw == 1:
                    kernels_1x1 += n_kernels
                elif kh == 3 and kw == 3:
                    kernels_3x3 += n_kernels
                else:
                    kernels_others += n_kernels

            linear = sum([
                l.weight.nelement() for l in self.model_list[i].modules() \
                if isinstance(l, nn.Linear)
            ])

            ckp.write_log(
                '1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.format(
                    kernels_1x1, kernels_3x3, kernels_others, linear
                ),
                refresh=True
            )
    def make_optimizer_all(self, ckp=None, lr=None):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.make_optimizer(i, ckp, lr))
        self.optimizer_list = ret
    
    def make_optimizer(self, i, ckp=None, lr=None):
        trainable = filter(lambda x: x.requires_grad, self.model_list[i].parameters())

        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum, 'nesterov': self.args.nesterov}

        kwargs['lr'] = self.args.lr if lr is None else lr
        kwargs['weight_decay'] = self.args.weight_decay
        # embed()
        optimizer = optimizer_function(trainable, **kwargs)

        if self.args.load != '' and ckp is not None:
            print('Loading the optimizer from the checkpoint...')
            optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )

        return optimizer
    def make_loss_all(self, Loss):
        self.loss_list = [Loss(self.args, self.ckp) for _ in self.fractions]
    
    def make_scheduler_all(self, resume=-1, last_epoch=-1, reschedule=-1):
        ret = []
        for s in self.optimizer_list:
            ret.append(self.make_scheduler(s, resume, last_epoch, reschedule))
        self.scheduler_list = ret

    def make_scheduler(self, target, resume=-1, last_epoch=-1, reschedule=0):
        if self.args.decay.find('step') >= 0:
            milestones = list(map(lambda x: int(x), self.args.decay.split('-')[1:]))
            kwargs = {'milestones': milestones, 'gamma': self.args.gamma}

            scheduler_function = lrs.MultiStepLR
            # embed()
            kwargs['last_epoch'] = last_epoch
            scheduler = scheduler_function(target, **kwargs)

        if self.args.load != '' and resume > 0:
            for _ in range(resume): scheduler.step()
        if reschedule>0:
            for _ in range(reschedule): scheduler.step()
        return scheduler
    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

