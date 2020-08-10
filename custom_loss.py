from torch import nn
import torch 

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (classes -1)
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, category):
        '''        
        category should be of same shape as pred indicating which categories the label could belong to
        '''
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing)
            true_dist = true_dist * ((category + torch.Tensor([1]).cuda()) * torch.Tensor([1.5]).cuda())
            true_dist[true_dist == 0] = self.smoothing
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))