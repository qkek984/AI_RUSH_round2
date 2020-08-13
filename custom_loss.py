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