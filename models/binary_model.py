import torch
from torch import nn

class Binary_Model(nn.Module):
    def __init__(self, model):
        super(Binary_Model, self).__init__()
        self.name = "Iterative Model"
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features,1)
        self.model.fc2 = nn.Linear(self.model.fc.in_features, 4)
        self.sigmoid = nn.Sigmoid()

        self.criterion_1 = nn.BCELoss()
        self.criterion_2 = nn.CrossEntropyLoss()
        
    def forward(self,x, oneh=None):

        x = self.model.feat_extract(x)
        if self.model.onehot:
            x = torch.cat([x, oneh], axis=-1)

        b_out = self.model.fc(x)
        b_out = b_out.squeeze(1)
        b_out = self.sigmoid(b_out)
        
        class_idx = (b_out < 0.5).nonzero().squeeze(1)
        unclass_idx = (b_out >= 0.5).nonzero().squeeze(1)

        class_feats = x[class_idx]
        class_out = self.model.fc2(class_feats)
        return b_out, class_out, unclass_idx, class_idx
