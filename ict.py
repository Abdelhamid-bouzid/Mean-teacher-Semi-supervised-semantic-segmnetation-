import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ICT(nn.Module):
    def __init__(self, alpha, T_model, ema_factor):
        super().__init__()
        self.alpha = alpha
        self.T_model = T_model
        self.T_model.train()
        self.ema_factor = ema_factor
        self.global_step = 0

    def forward(self, x, y, S_model, mask):
        # NOTE: this implementaion uses mixup for only unlabeled data
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step += 1 # for moving average coef
        
        mask = mask.byte()
        S_model.update_batch_stats(False)
        mt_y = self.T_model(x.cpu()).detach()
        u_x, u_y = x[mask], mt_y[mask]
        l_x, l_y = x[mask==0], mt_y[mask==0]
        
        lam  = np.random.beta(self.alpha, self.alpha) # sample mixup coef
        perm = torch.randperm(u_x.shape[0])
        perm_u_x, perm_u_y = u_x[perm], u_y[perm]
        
        mixed_u_x = lam * u_x + (1 - lam) * perm_u_x
        mixed_u_y = (lam * u_y + (1 - lam) * perm_u_y).detach()
        
        y_hat = S_model(torch.cat([l_x, mixed_u_x], 0)) # "cat" indicates to compute batch stats from full batches
        loss = F.mse_loss(y_hat.softmax(1), torch.cat([l_y, mixed_u_y], 0).softmax(1).to(device=device), reduction="none").mean([1,2,3])
        # compute loss for only unlabeled data, but loss is normalized by full batchsize
        loss = loss[l_x.shape[0]:].sum() / x.shape[0]
        S_model.update_batch_stats(True)
        return loss

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step), self.ema_factor)
        for emp_p, p in zip(self.T_model.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
