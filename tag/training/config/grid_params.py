import torchvision
import PIL
import torch
from .autoaugment import ImageNetPolicy

# imagenet
rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

# dataset 
# rgb_mean = [0.55232704, 0.51815085, 0.48528248]
# rgb_std =  [0.21313286, 0.21373375, 0.21965458]

grid = {
    'batch_size' : [128],
    'class_ratio' : [
            [1, 0.2, 0.3, 0.9],
            [1, 0.2, 0.3, 0.5]            
        ],
    'epoch' : [
            [3,10], [0,15]
        ],
    'gamma' : [0.3],
    'learning_rate' : [1e-4, 1e-3],
    'loss_cls_weight' : [
            torch.Tensor([1,1,1,1.1]),
            torch.Tensor([1,1,1,1.2])
        ],
    'optimizer' : ['Adam'], # 
    'step_size': [5, 2],
    'transforms' : [
            [3, torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ColorJitter(hue=.1, saturation=.1),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])]
        ],
    'val_ratio' : [
            [0.025, 0.2, 0.15, 0.15]
        ],
    'weight_decay' : [1e-3]
}