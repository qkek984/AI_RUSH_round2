import torch
from torch import nn

class Trainable_Embedding(nn.Module):
    def __init__(self, model, embed_dim=90):
        super(Trainable_Embedding, self).__init__()
        self.name = "Trainable Embedding"
        self.model = model
        self.embed_dim = embed_dim
        self.cat_embedding = nn.Parameter(torch.randn(9, embed_dim), requires_grad=True).cuda()
        if "ResNet" in self.model.name:     
            self.fc = nn.Linear(2048 + embed_dim, 5).cuda()
        elif "ResNext" in self.model.name:     
            self.fc = nn.Linear(1000 + embed_dim, 5).cuda()

    def forward(self, x, category):

        x, _ = self.model(x)
        
        with torch.no_grad():
            x = torch.cat([x, torch.gather(self.cat_embedding,0, category.repeat(1,int(self.embed_dim/9)).long())], axis=1)
        x = self.fc(x)
        pred = torch.argmax(x, dim=-1)
        return x , pred

