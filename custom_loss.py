from torch import nn
import torch 

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing, attention, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (classes -1)
        self.attention = attention
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, category):
        '''        
        category should be of same shape as pred indicating which categories the label could belong to
        '''
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():

            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing)
            category[category == 1] =  self.attention
            category[category == 0] =  1
            true_dist = true_dist * category
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

    def __repr__(self):
        weights = [self.confidence , self.smoothing * self.attention, self.smoothing]
        return "LabelSmoothingLoss(nn.Module)" +" with weight " + str([int(w/sum(weights) * 100) for w in weights]) 

class AlphaCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0, loss_fcn= None):
        super(AlphaCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        if loss_fcn:
            self.loss_fcn = loss_fcn
        else:
            self.loss_fcn = nn.CrossEntropyLoss(reduction='mean')

        self.loss_fcn_2 = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, pred, target, pseudo_target, category_pos=None):
        if isinstance(self.loss_fcn,nn.CrossEntropyLoss):
            if self.alpha:
                og_loss = self.loss_fcn_2(pred, target) * (1 - self.alpha)
                correct_loss = self.loss_fcn_2(pred, pseudo_target) * self.alpha
                return  og_loss + correct_loss 
            else:
                og_loss = self.loss_fcn(pred, target) 
                return  og_loss 

        else:
            if self.alpha:
                og_loss = self.loss_fcn(pred, target, category_pos) * (1 - self.alpha)
                correct_loss = self.loss_fcn(pred, pseudo_target, category_pos) * self.alpha
                return  og_loss + correct_loss 
            else:
                og_loss = self.loss_fcn(pred, target, category_pos)                
                return  og_loss 


