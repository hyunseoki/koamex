import sys
import tqdm
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, loss_func, device, optimizer):
    running_metric = {
        'loss' : AverageMeter(),
    }
  
    model.train()       

    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for image, targets in iterator:
            image = [i.to(device) for i in image]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss = model(image, targets)
            loss = sum(loss for loss in loss.values())

            running_metric['loss'].update(loss.item(), image.size(0))

            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
                  
            log = 'loss - {:.5f}'.format(running_metric['loss'].avg)
            iterator.set_postfix_str(log)

    return running_metric['loss'].avg

def validate(valid_loader, model, loss_func, device):
    pass


def save_checkpoint(state, save_dir, fn='model_best.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = os.path.join(save_dir, fn)
    torch.save(state, save_fn)
    print(f'MODEL IS SAVED TO {save_fn}!!!')


class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_func, optimizer, device, save_dir, 
                       mode='max', scheduler=None, num_epochs=25):

        assert mode in ['min', 'max']

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_dir
        self.mode = mode
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        self.elapsed_time = None

        self.log = {
            'train_loss' : list(),
            'valid_loss' : list(),
        }

        self.lr_curve = list()

    def train(self):
        """fit a model"""

        if self.device == 'cpu':
            print('[info msg] Start training the model on CPU')
        else:
            print(f'[info msg] Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print('=' * 50)              
        
        if self.mode =='max':
            best_metric = -float('inf')
        else:
            best_metric = float('inf')

        self.model = self.model.to(self.device)
        startTime = datetime.now()     

        print('[info msg] training start !!')
        for epoch in range(self.num_epochs):        
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            train_loss = train(
                train_loader=self.train_loader,
                model=self.model,
                loss_func=self.loss_func,
                device=self.device,
                optimizer=self.optimizer,
                )
            self.log['train_loss'].append(train_loss)

        self.elapsed_time = datetime.now() - startTime
        self.__save_result()

    def __save_result(self):    
        pass

    @property
    def save_dir(self):
        return self.save_path