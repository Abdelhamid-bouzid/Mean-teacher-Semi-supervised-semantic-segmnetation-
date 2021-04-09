# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:23:32 2021

@author: Admin
"""

from loss_function import loss_function
import torch
import torch.nn.functional as F
from config import config
from IOU import iou_pytorch,iou_numpy
import numpy as np
from torch.utils.data import DataLoader
from RandomSampler import RandomSampler
import math

def learning_function(S_model,ssl_obj,l_train,test, u_train):
    
    
    ''' #################################################  set up optim  ################################################### '''
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_model.train()
    
    optimizer     = torch.optim.Adam(S_model.parameters(),lr = config['learning_rate'])
    
    ''' #################################################  Dtat loaders  ################################################### '''
    train_sampler=RandomSampler(len(l_train), config["iteration"] * config["batch_size"]//2)
    l_loader = DataLoader(l_train, config["batch_size"]//2,drop_last=True,sampler=train_sampler)
    
    u_sampler=RandomSampler(len(u_train), config["iteration"] * config["batch_size"]//2)
    u_loader = DataLoader(u_train, config["batch_size"]//2,drop_last=True,sampler=u_sampler)
    
    test_loader = DataLoader(test, config["batch_size"],drop_last=False)
    
    ''' #################################################  initialization  ################################################### '''
        
    Loss_train,Loss_test,train_ious,test_ious = [],[],[],[]
    best_iou  = 0
    iteration = 0
    for l_data, u_data in zip(l_loader, u_loader):
        S_model     = S_model.to(device=device, dtype=torch.float)
        iteration += 1
        coef       = config["consis_coef"] * math.exp(-5 * (1 - min(iteration/config["warmup"], 1))**2)
        
        l_input, l_target = l_data
        l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
        
        u_input, _ = u_data
        u_input    = u_input.to(device).float()
        
        inputs = torch.cat([l_input, u_input], 0)
        
        m = torch.cat([torch.ones((l_target.shape[0],)), -1*torch.ones((u_input.shape[0],))])
        unlabeled_mask = (m == -1).float()
        
        outputs = S_model(inputs)
        
        cls_loss = loss_function().forward(outputs[:l_target.shape[0]], l_target)
        ssl_loss = ssl_obj(inputs, outputs.detach(), S_model, unlabeled_mask) * coef
        
        loss = cls_loss + ssl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ssl_obj.moving_average(S_model.cpu().parameters())
        
        iou = iou_pytorch(outputs[:l_target.shape[0]].argmax(dim=1).cpu(),l_target.argmax(dim=1).cpu())
        train_ious.append(iou)
        
        #print(cls_loss.item(),ssl_loss.item(),coef,iou)
        print('##########################################################################################################')
        print("   #####  Train iteration: {} train_loss: {:0.4f} train_iou: {:0.4f}".format(iteration,loss.item(),iou))
        print('##########################################################################################################')
        
        if iteration%config["test_model_cycel"]==0:
            S_model     = S_model.to(device=device, dtype=torch.float)
            S_model.eval()
            test_iou  = 0
            for l_input, l_target in test_loader:
                l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
                
                outputs = S_model(l_input)
                
                iou = iou_pytorch(outputs.argmax(dim=1).cpu(),l_target.argmax(dim=1).cpu())
                test_iou += iou
                    
            test_iou = test_iou/len(test)
        
            print('**********************************************************************************************************')
            print("   #####  Train iteration: {} test_iou: {:0.4f} ".format(iteration,iou))
            print('**********************************************************************************************************')
        
            if test_iou>best_iou:
                best_iou = test_iou
                torch.save(ssl_obj.T_model,'models/model.pth')
            
        
        S_model.train()
       
    return train_ious,test_ious