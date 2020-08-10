import torch
from torch import nn

class Trainable_Embedding(nn.Module):
    def __init__(self, model):
        super(Trainable_Embedding, self).__init__()
        self.name = "Trainable Embedding"
        self.model = model
        self.cat_embedding = nn.Parameter(torch.Tensor([[0] * 50 ] * 9).cuda(), requires_grad=True)
        if "ResNet" in self.model.name:     
            self.fc = nn.Linear(2048 + 50, 5)

    def forward(self, x, category):

        x, _ = self.model(x)
        x = torch.cat([x, torch.gather(cat_embedding,0, category.repeat(1,9).long())], axis=1)
        x = self.fc(x)

        return x 

