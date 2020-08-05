from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def EfficientNet_B3(pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b3')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'blocks.24' not in name and 'blocks.25' not in name:
                param.requires_grad = False
    else:
        model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Linear(1536, 5)    
    print("EfficientNet B3 Loaded!")

    return model