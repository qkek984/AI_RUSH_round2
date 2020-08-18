import torch
from torch import nn
import torch.nn.functional as F

class Binary_Model(nn.Module):
    def __init__(self, model, cat_embed=0, embed_dim=45):
        super(Binary_Model, self).__init__()
        self.name = "Binary Model"
        self.model = model
        self.cat_embed = cat_embed
        self.embed_dim = embed_dim
        self.onehot = 1
        if self.cat_embed:
            self.onehot = 0
            self.cat_embedding = nn.Parameter(torch.randn(9, embed_dim), requires_grad=True)        
        in_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feat - self.model.onehot * self.cat_embed * 9  + self.cat_embed * embed_dim,1)
        self.model.fc2 = nn.Linear(in_feat -  self.model.onehot * self.cat_embed * 9 + self.cat_embed * embed_dim, 4)
        self.sigmoid = nn.Sigmoid()

        self.criterion_1 = nn.BCELoss()
        self.criterion_2 = nn.CrossEntropyLoss()
        
    def forward(self,x,oneh=None, category=None):

        x = self.model.feat_extract(x)
        if self.cat_embed:
            x = torch.cat([x, torch.gather(self.cat_embedding,0, category.repeat(1,self.embed_dim).long())], axis=1)
        
        if self.model.onehot or self.model.onehot2:
            x = torch.cat([x, oneh], axis=-1)
        
        b_out = self.model.fc(x)
        b_out = b_out.squeeze(1)
        b_out = self.sigmoid(b_out)
        
        class_idx = (b_out < 0.5).nonzero().squeeze(1)
        unclass_idx = (b_out >= 0.5).nonzero().squeeze(1)

        class_feats = x
        class_out = F.softmax(self.model.fc2(class_feats))

        return b_out, class_out, unclass_idx, class_idx
