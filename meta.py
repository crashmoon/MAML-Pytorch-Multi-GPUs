""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from samplers import CategoriesSampler, GraphSampler
from meta_learner import MetaLearner
from misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataset_loader import DatasetLoader as Dataset
import plot
import json
import yaml
from torch import optim
from copy import deepcopy
from collections import deque

class MetaTrainer(object):    
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'maml'])
        save_path2 = (
            'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) 
            +'_lr' + str(args.meta_lr) +'_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) 
            +'_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step)             
            + '_' + args.meta_label 
            )
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.args = args
        self.trainset = Dataset('train', self.args, train_aug = True)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
        #self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)
        self.train_loader = None
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, self.args.val_batch, self.args.way, self.args.shot + self.args.val_query)
        #self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        self.val_loader = None
        self.model = MetaLearner(self.args).to(self.args.device)
        ##self.model.encoder.load_state_dict(torch.load(self.args.pre_load_path))
        self.model = torch.nn.DataParallel(self.model)        
        print(self.model)                
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.args.meta_lr, momentum = 0.9, weight_decay=args.weight_decay)   # or adam
        

    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           


    def inf_get(self, train):
        while (True):
            for x in train:
                yield x[0]


    def confidence_func(self, acc, temperature = 1.0):
        return acc ** temperature


    def train(self):
        """The function for the meta-train phase."""
        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        timer = Timer()
        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot).to(self.args.device).type(torch.long)
        label_query = torch.arange(self.args.way).repeat(self.args.train_query).to(self.args.device).type(torch.long)
        
        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            ################### train #############################
            self.model.train()
            train_acc_averager = Averager()            
            self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=self.args.num_work, pin_memory=True)
            train_data = self.inf_get(self.train_loader)
            self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=self.args.num_work, pin_memory=True)
            val_data = self.inf_get(self.val_loader)
            acc_log = []
            tqdm_gen = tqdm.tqdm(range(self.args.num_batch//self.args.meta_batch))
            for i in tqdm_gen:
                data_list = []
                label_shot_list = []
                for _ in range(self.args.meta_batch):
                    data_list.append(train_data.__next__().to(self.args.device))
                    label_shot_list.append(label_shot)
                pass
                data_list = torch.stack(data_list, dim=0)
                label_shot_list = torch.stack(label_shot_list, dim=0)   # shot-label
                out = self.model(data_list, label_shot_list)                
                meta_loss = 0
                for inner_id in range(self.args.meta_batch):                                        
                    meta_loss += F.cross_entropy(out[inner_id], label_query) 
                    cur_acc = count_acc(out[inner_id], label_query)
                    acc_log.append( (i*self.args.meta_batch+inner_id, cur_acc) )
                    train_acc_averager.add(cur_acc)
                    tqdm_gen.set_description('Epoch {}, Acc={:.4f}'.format(epoch, cur_acc))
                pass
                meta_loss /= self.args.meta_batch
                plot.plot('meta_loss', meta_loss.item())
                self.optimizer.zero_grad()
                meta_loss.backward()
                self.optimizer.step()
                plot.tick()
            pass
            train_acc_averager = train_acc_averager.item()
            trlog['train_acc'].append(train_acc_averager)
            plot.plot('train_acc_averager', train_acc_averager)
            plot.flush(self.args.save_path)                        

            ####################### eval ##########################
            #self.model.eval()    ###############################################################################
            val_acc_averager = Averager()
            for i in tqdm.tqdm(range(self.args.val_batch//self.args.meta_batch)):
                data_list = []
                label_shot_list = []
                for _ in range(self.args.meta_batch):
                    data_list.append(val_data.__next__().to(self.args.device))
                    label_shot_list.append(label_shot)
                pass
                data_list = torch.stack(data_list, dim = 0) 
                label_shot_list = torch.stack(label_shot_list, dim = 0)
                out = self.model(data_list, label_shot_list).detach()
                for inner_id in range(self.args.meta_batch):
                    cur_acc = count_acc(out[inner_id], label_query)
                    val_acc_averager.add(cur_acc)
                pass
            pass
            val_acc_averager = val_acc_averager.item()
            trlog['val_acc'].append(val_acc_averager)
            print('Epoch {}, Val, Acc={:.4f}'.format(epoch, val_acc_averager))
            
            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')        

            with open(osp.join(self.args.save_path, 'trlog.json'), 'w') as f:
                json.dump(trlog, f)
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        pass
        
        
    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        with open(osp.join(self.args.save_path, 'trlog.json'), 'r') as f:
            trlog = yaml.load(f)

        # Load meta-test set
        test_set = Dataset('test', self.args, train_aug = False)
        sampler = CategoriesSampler(test_set.label, self.args.test_batch, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=self.args.num_work, pin_memory=True)
        test_data = self.inf_get(loader)

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        #self.model.eval()    ################################ ????????? ################################################################

        # Set accuracy averager
        ave_acc = Averager()
        acc_log = []

        # Generate labels
        label_shot = torch.arange(self.args.way).repeat(self.args.shot).to(self.args.device).type(torch.long)
        label_query = torch.arange(self.args.way).repeat(self.args.train_query).to(self.args.device).type(torch.long)
            

        for i in tqdm.tqdm(range(self.args.test_batch//self.args.meta_batch)):
            data_list = []
            label_shot_list = []
            for _ in range(self.args.meta_batch):
                data_list.append(test_data.__next__().to(self.args.device))
                label_shot_list.append(label_shot)
            pass
            data_list = torch.stack(data_list, dim = 0)
            label_shot_list = torch.stack(label_shot_list, dim = 0)
            out = self.model(data_list, label_shot_list).detach()
            for inner_id in range(self.args.meta_batch):
                cur_acc = count_acc(out[inner_id], label_query)
                acc_log.append(cur_acc)
                ave_acc.add(cur_acc)
            pass
        pass
        
        acc_np = np.array(acc_log, dtype=np.float)
        m, pm = compute_confidence_interval(acc_np)


        trlog['test_acc'] = [m, pm]
        cur_test_save_name = 'trlog_test_' + str(self.args.index) + '.json'
        with open(osp.join(self.args.save_path, cur_test_save_name), 'w') as f:
            json.dump(trlog, f)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
