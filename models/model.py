import torch
import torch.nn.functional as F
import torchvision.models as models


class Resnet50_FC2(torch.nn.Module):
    def __init__(self, n_class=5, pretrained=True):
        super(Resnet50_FC2, self).__init__()
        self.basemodel = models.resnet50(pretrained=pretrained)
        self.linear1 = torch.nn.Linear(1000 + 9, 512)
        self.linear2 = torch.nn.Linear(512, n_class)

    def forward(self, x, onehot):
        x = self.basemodel(x)
        x = torch.cat([x,onehot], axis=1)
        
        x = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(x), dim=-1)
        pred = torch.argmax(out, dim=-1)
        return out, pred