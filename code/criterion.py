import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(setting, device, class_weight=None):
    if setting.criterion == 'CrossEntropyLoss':
        main_criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    elif setting.criterion == 'F1_Loss':
        main_criterion = F1_Loss().to(device)
    else:
        raise NotImplementedError(f'No criterion support for {setting.criterion}')

    if setting.model_type in ['gate', 'mixed_gate']:
        gate_criterion = nn.BCEWithLogitsLoss().to(device)
            
        return main_criterion, gate_criterion
    else:
        return main_criterion


class F1_Loss(nn.Module):
    '''Calculate Multi Class F1 score
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''
    def __init__(self, epsilon=1e-7, total_class=46):
        super().__init__()
        self.epsilon = epsilon
        self.total_class = 46
    
    def forward(self, y_pred, y_true):
        res = []

        for ci in range(self.total_class):
            prob_ci = y_pred[:, ci]

            tp = ((y_true == ci) * prob_ci).sum(dim=-1)
            fp = ((y_true != ci) * prob_ci).sum(dim=-1)
            fn = ((y_true == ci) * (1-prob_ci)).sum(dim=-1)

            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)
            f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
            res.append(f1)
        
        return 1 - torch.stack(res, dim=-1).mean()