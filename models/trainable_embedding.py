import torch
from torch import nn

class Trainable_Embedding(nn.Module):
    def __init__(self, model, embed_dim=90):
        super(Trainable_Embedding, self).__init__()
        self.name = "Trainable Embedding"
        self.model = model
        self.onehot = 0
        self.onehot2= model.onehot2
        self.embed_dim = embed_dim
        self.cat_embedding = nn.Parameter(torch.randn(9, embed_dim), requires_grad=True)
        self.fc = nn.Linear(self.model.fc.in_features + embed_dim - self.model.onehot * 9, 5)

    def forward(self, x, category, onehot=None):
        x = self.model.feat_extract(x)
        x = torch.cat([x, torch.gather(self.cat_embedding,0, category.repeat(1,self.embed_dim).long())], axis=1)
        if self.model.onehot2:
            x = torch.cat([x, onehot], axis=1)
        x = self.fc(x)
        pred = torch.argmax(x, dim=-1)
        return x , pred

