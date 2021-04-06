# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:41:38 2021

@author: Admin
"""
import torch
from torch.autograd import Function,Variable

class loss_function(Function):
    def __init__(self):
        self.loss   = torch.nn.CrossEntropyLoss()
        self.SMOOTH = 1e-6
    
    def forward(self, pred, truth):
        loss = self.loss(pred, truth.argmax(dim=1))
        

# =============================================================================
#         intersection = (pred.argmax(1) & truth.argmax(1)).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#         union = (pred.argmax(1) | truth.argmax(1) ).float().sum((1, 2))         # Will be zzero if both are 0
#         
#         iou = (intersection + self.SMOOTH) / (union + self.SMOOTH)  # We smooth our devision to avoid 0/0
#         loss = 1 - iou.mean()
#         loss = Variable(loss, requires_grad = True)
# =============================================================================
        return loss