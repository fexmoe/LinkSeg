import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch_geometric.nn import dense_mincut_pool
from sklearn.utils.class_weight import compute_class_weight


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()
    
    def forward(self, predictions, gt):
        gt = gt.squeeze()
        predictions = predictions.squeeze()
        num = 2*((predictions*gt).sum())
        denom = predictions.pow(2).sum() + gt.pow(2).sum() 
        return 1 - num / denom
    


class Class_loss(nn.Module):
    def __init__(self, smoothing=0):
        super(Class_loss, self).__init__()
        if smoothing == 0:
            self.ce = torch.nn.CrossEntropyLoss() 
        else:
            self.ce = torch.nn.CrossEntropyLoss(label_smoothing=smoothing) 

    def forward(self, predictions, gt, balance=False):
        gt_numpy = gt.view(-1).cpu().numpy()
        nb_classes = predictions.size(-1)

        if balance:
            class_weights=compute_class_weight(class_weight="balanced", classes=np.unique(gt_numpy), y=gt_numpy)
            class_weights=torch.tensor(class_weights,dtype=torch.float, device=predictions.device)
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)

        loss = self.ce(predictions.view(-1, nb_classes), gt.view(-1), )
        pred_numpy = predictions.view(-1, nb_classes)
        pred_numpy = torch.argmax(pred_numpy, -1)
        pred_numpy = pred_numpy.cpu().detach().numpy()
        acc = balanced_accuracy_score(gt_numpy, pred_numpy)
        return loss, acc
    


class MIN_CUT_loss(nn.Module):
    def __init__(self):
        super(MIN_CUT_loss, self).__init__()
    
    def forward(self, x, s, adj=None):
        x, adj, mc, o = dense_mincut_pool(x, adj, s)
        return mc, o
    

def get_losses():
    return {'dice_loss' : Dice_loss(), 
            'class_loss_ssm' : Class_loss(), 
            'mincut_loss' : MIN_CUT_loss(), 
            'class_loss_section' : Class_loss()}