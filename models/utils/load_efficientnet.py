from models.efficientnet import EfficientNet
import torch.nn as nn

def EfficientNet_B5(pretrained=True, advprop=False, onehot=0):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b5', onehot=onehot)
        for name, param in model.named_parameters():
            if 'fc' not in name :# and 'blocks.24' not in name and 'blocks.25' not in name
                param.requires_grad = False
    else:
        model = EfficientNet.from_name('efficientnet-b5', onehot=onehot)
    model.fc = nn.Linear(2048 + onehot * 9, 5)
    model.name = "EfficientNet_B5"    
    print("EfficientNet B5 Loaded!")

    return model

def EfficientNet_B7(pretrained=True, advprop=False, onehot=0):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b7', onehot=onehot,advprop=advprop)
        for name, param in model.named_parameters():
            if 'fc' not in name :# and 'blocks.24' not in name and 'blocks.25' not in name
                param.requires_grad = False
    else:
        model = EfficientNet.from_name('efficientnet-b7', onehot=onehot, advprop=advprop)
    model.fc = nn.Linear(2560 + onehot * 9, 5)    
    model.name = "EfficientNet_B7"    
    print("EfficientNet B7 Loaded!")

    return model

def EfficientNet_B8(pretrained=True, onehot=0):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b8', onehot=onehot, advprop=True)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    else:
        model = EfficientNet.from_name('efficientnet-b8', onehot=onehot, advprop=True)
    model.fc = nn.Linear(2816 + onehot * 9 , 5)    
    model.name = "EfficientNet_B8"    
    print("EfficientNet B8 Loaded!")

    return model