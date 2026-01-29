# imports
import torch
import torch.nn as nn

# loss function (-si_snr)
class loss_fn(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predictions, targets):
        # normalize preds and targets to zero-mean
        predictions = predictions - torch.mean(predictions, dim = -1, keepdim = True)
        targets = targets - torch.mean(targets, dim = -1, keepdim = True)

        scaling_factor = torch.sum(predictions * targets, dim = -1, keepdim = True) / torch.sum(targets**2, dim = -1, keepdim = True)
        scaled_targets = scaling_factor * targets
        noise = predictions - targets
        mean_si_snr = torch.mean(10 * torch.log10((torch.sum(scaled_targets**2, dim = -1) + 1e-8) / (torch.sum(noise**2, dim = -1)) + 1e-8), dim = -1)
        loss = -torch.mean(mean_si_snr) # average loss across batch
        return loss