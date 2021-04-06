import torch
import torch.nn as nn
import torch.nn.functional as F

class MT(nn.Module):
    def __init__(self, T_model, ema_factor):
        super().__init__()
        self.T_model = T_model
        self.T_model.train()
        self.ema_factor = ema_factor
        self.global_step = 0

    def forward(self, x, y, S_model, mask):
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step += 1
        y_hat = self.T_model(x.cpu())
        S_model.update_batch_stats(False)
        y = S_model(x) # recompute y since y as input of forward function is detached
        S_model.update_batch_stats(True)
        
        loss = (F.mse_loss(y.softmax(1), y_hat.softmax(1).to(device=device), reduction="none").mean([1,2,3]) * mask.to(device=device)).mean()
        
        return loss

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
        for emp_p, p in zip(self.T_model.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
