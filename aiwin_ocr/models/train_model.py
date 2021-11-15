import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import wandb
import models
from models.ctc import ctc_decode

class Trainer(object):
    """
    """
    def __init__(self,config:dict,logger:logging,labels2char, eval = False):
        super().__init__(), 
        self.config = config
        self.val_epochs = []
        self.train_epochs = []
        self.acc = []
        self.val_loss = []
        self.f1_score = []
        self.train_loss = []
        self.net = getattr(models,config['base_model'])(config)
        self.device = self.config['base']['device']
        self.net.to(self.device)
        self.early_stop = config['train']['early_stop']
        self.best_score = 0.0
        self.early_stop_count = 0
        self.eval = eval 
        self.logger = logger
        self.labels2char = labels2char
        if not self.eval:
            self.experiment = wandb.init(self.net.name())
        self.load()
    
    def parameters(self):
        return self.net.parameters()
    
    def cal_acc(self, preds, reals, target_lengths):
        tot_correct = 0
        wrong_cases = []
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            if real == pred:
                tot_correct += 1
            else:
                wrong_cases.append((real, pred))
        return tot_correct
    
    def val(self, criterion,epoch, data_loader):
        self.net.eval()
        self.val_epochs.append(epoch)
        total_loss = 0
        total_count = 0.0
        total_acc = 0
        pbar = tqdm(total=len(data_loader), desc=f"Eval, epoch:{epoch}")
        with torch.no_grad():
             for data in data_loader:
                 images, targets, target_lengths = [d.to(self.device) for d in data]
                 out = self.net(images)
                 seq_len, batch, num_class = out.size()
                 log_probs = F.log_softmax(out, dim=2)
                 input_lengths = torch.LongTensor([seq_len] * batch)
                 loss = criterion(log_probs, targets, input_lengths, target_lengths)
                 preds = ctc_decode(log_probs, method=self.config['loss']['decode_method'], 
                      beam_size=self.config['loss']['beam_size'])
                 reals = targets.cpu().numpy().tolist()
                 total_loss += loss.item()
                 total_count += batch
                 target_lengths = target_lengths.cpu().numpy().tolist()
                 total_acc += self.cal_acc(preds,reals,target_lengths)
                 pbar.update(1)
             pbar.close()
             self.val_loss.append(total_loss/total_count)
             self.acc.append(total_acc/total_count)
             self.experiment.log({
                'val loss': self.val_loss[-1],
                'val acc':self.acc[-1],
                'epoch': epoch,
                'images': wandb.Image(images[0].cpu(),caption=f'Real:{self.decode_target(reals[0:target_lengths[0]])}, Pred:{self.decode_target(preds[0])}'),
              })
             if self.best_score < self.acc[-1]:
                self.best_score = self.acc[-1]
                self.early_stop_count = 0
                self.save()
             else:
                self.early_stop_count += 1            

    def train(self, optimizer, scheduler,criterion, epoch, data_loader):
        self.net.train()
        self.train_epochs.append(epoch)
        total_loss = 0.0
        total_count = 0
        pbar = tqdm(total=len(data_loader), desc=f"Train, epoch:{epoch}")
        for data in data_loader:
            images, targets, target_lengths = [d.to(self.device) for d in data]
            out = self.net(images)
            seq_len, batch, num_class = out.size()
            log_probs = F.log_softmax(out, dim=2)
            input_lengths = torch.LongTensor([seq_len] * batch)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += batch
            pbar.update(1)
        pbar.close()
        scheduler.step()
        self.train_loss.append(total_loss/total_count)
        self.experiment.log({
           'train loss':self.train_loss[-1],
           'epoch': epoch,
           'lr': scheduler.get_last_lr()[0]
        })
    
    def save(self):
        save_path = self.net.save()
        self.logger.info(f'save model at {save_path}, epoch:{self.train_epochs[-1]}, loss:{self.val_loss[-1]},acc:{self.acc[-1]} ')
    
    
    def info(self):
        self.logger.info(f'Epoch {self.train_epochs[-1]}, Train loss {self.train_loss[-1]}, Val loss {self.val_loss[-1]} Val acc {self.acc[-1]} ')
    
    def runing(self):
        return self.early_stop_count < self.early_stop
    
    def load(self):
        if self.net.load():
            self.logger.info(f"{self.net.name()} load model's parameters from {self.config['base']['load_path']}")
    
    def predict(self,test_load):
        self.net.eval()
        all_preds = []
        all_reals = []
        with torch.no_grad():
            for data in tqdm(test_load):
                images, targets, target_lengths = [d.to(self.device) for d in data]
                out = self.net(images)
                log_probs = F.log_softmax(out, dim=2)
                preds = ctc_decode(log_probs, method=self.config['loss']['decode_method'], 
                      beam_size=self.config['loss']['beam_size'])
                target_lengths = target_lengths.cpu().numpy().tolist()
                reals = targets.cpu().numpy().tolist()
                target_length_counter = 0
                for pred,target_length in zip(preds,target_lengths):
                    real = reals[target_length_counter:target_length_counter + target_length]
                    target_length_counter += target_length
                    all_preds.append(self.decode_target(pred))
                    all_reals.append(self.decode_target(real))
        return all_reals,all_preds
    
    def decode_target(self,sequence):
        return ''.join([self.labels2char[x] for x in sequence]).replace(' ', '')

