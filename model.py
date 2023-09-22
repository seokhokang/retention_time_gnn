import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:

    def __init__(self, net, source_path, cuda):
    
        self.net = net.to(cuda)
        self.source_path = source_path
        self.target_path = ''
        self.cuda = cuda


    def load(self, model_path):
      
        self.net.load_state_dict(torch.load(model_path)) 
        
               
    def training(self, train_loader, val_loader, mean, std, method, opt, max_epochs = 500, verbose = False):
   
        # transfer learning method:
        if method == 'scratch':
            pass
        elif method == 'finetune':
            self.load(self.source_path)#load source model
            for layer in self.net.predict:#reset prediction head
               if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        elif method == 'feature':
            self.load(self.source_path)#load source model
            for param in self.net.gnn.parameters(): param.requires_grad = False
            for layer in self.net.predict:#reset prediction head
               if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
               
        print('no. trainable parameters', sum(p.numel() for p in self.net.parameters() if p.requires_grad))
  
        loss_fn = nn.HuberLoss()
        
        if opt == 'adam':
            
            optimizer = Adam(self.net.parameters(), lr = 1e-4, weight_decay = 1e-5)
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6, verbose=True)    
    
            if hasattr(val_loader.dataset, 'label'):
                train_size = train_loader.dataset.__len__()
                val_size = val_loader.dataset.__len__()
                val_y = val_loader.dataset.label
            else:
                train_size = train_loader.dataset.indices.__len__()
                val_size = val_loader.dataset.indices.__len__()
                val_y = val_loader.dataset.dataset.label[val_loader.dataset.indices]  
            
                
            val_log = np.zeros(max_epochs)
            for epoch in range(max_epochs):
                
                # training
                self.net.train()
                start_time = time.time()
    
                for batchidx, batchdata in enumerate(train_loader):
        
                    inputs = [b.to(self.cuda) for b in batchdata[:-1]]
                    labels = ((batchdata[-1] - mean) / std).to(self.cuda)
                    
                    preds = self.net(*inputs)
                    loss = loss_fn(preds, labels)
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss = loss.detach().item()
        
                if verbose:
                    print('--- training epoch %d, lr %f, processed %d, loss %.3f, time elapsed(min) %.2f'
                      %(epoch + 1, optimizer.param_groups[-1]['lr'], train_size, train_loss, (time.time()-start_time)/60))
        
                # validation
                start_time = time.time()
                val_y_preds = self.inference(val_loader, mean, std) 
    
                val_loss = loss_fn(torch.FloatTensor((val_y_preds - mean)/std), torch.FloatTensor((val_y - mean)/std)).detach().cpu().numpy()
                lr_scheduler.step(val_loss)
                val_log[epoch] = val_loss
        
                if verbose:
                    print('--- validation at epoch %d, processed %d, loss %.3f (BEST %.3f), monitor %d, time elapsed(min) %.2f'
                      %(epoch + 1, val_size, val_loss, np.min(val_log[:epoch + 1]), epoch - np.argmin(val_log[:epoch + 1]), (time.time()-start_time)/60))  
        
                # save minimun-loss parameters
                if np.argmin(val_log[:epoch + 1]) == epoch:
                    torch.save(self.net.state_dict(), self.target_path) 

                # earlystopping
                elif np.argmin(val_log[:epoch + 1]) <= epoch - 30:
                    break
            
            print('training terminated at epoch %d' %(epoch + 1))
            self.load(self.target_path)

        elif opt == 'lbfgs':
    
            def loss_calc():
            
                preds = self.net(*inputs)
                loss = loss_fn(preds, labels)
                loss += 1e-5 * torch.stack([p.square().sum() for p in self.net.parameters()]).sum()
                
                return loss
    
            def closure():
            
                optimizer.zero_grad()
                loss = loss_calc()
                loss.backward()
    
                return loss

            # load data
            for batchidx, batchdata in enumerate(train_loader):
         
                inputs = [b.to(self.cuda) for b in batchdata[:-1]]
                labels = ((batchdata[-1] - mean) / std).to(self.cuda)
                break
    
            # training details
            optimizer = LBFGS(self.net.parameters(), lr = 1, max_iter = 1)

            val_log = np.zeros(max_epochs)
            for epoch in range(max_epochs):
            
                self.net.train()
                optimizer.step(closure)
    
                val_log[epoch] = loss_calc().detach().cpu().numpy()
                if np.isnan(val_log[epoch]): val_log[epoch] = 1e5
                
                # learning rate decay
                optimizer.param_groups[0]['lr'] -= 1/max_epochs 
    
                # save minimun-loss parameters
                if np.argmin(val_log[:epoch + 1]) == epoch:
                    torch.save(self.net.state_dict(), self.target_path) 
    
            print('training terminated at iter %d'%(np.argmin(val_log) + 1))
            self.load(self.target_path)
                
    
    def inference(self, tst_loader, mean, std):
                 
        self.net.eval()
        tst_y_preds = []
        with torch.no_grad():
            for batchidx, batchdata in enumerate(tst_loader):
            
                inputs = [b.to(self.cuda) for b in batchdata[:-1]]

                preds_list = self.net(*inputs).cpu().numpy()
                tst_y_preds.append(preds_list)
    
        tst_y_preds = np.vstack(tst_y_preds) * std + mean
    
        return tst_y_preds
