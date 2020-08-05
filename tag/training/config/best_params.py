import torchvision
import PIL
import torch
from .autoaugment import ImageNetPolicy

rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

best_param = {
    'batch_size' : [128],
    'class_ratio' : [
            [1, 0.2, 0.3, 0.8, 0.5]
            # [0.0001,0.0001,0.0001,0.0001, 0.0001]            
        ],
    'epoch' : [
            [10,10]
        ],
    'gamma' : [0.3],
    'learning_rate' : [1e-5],
    'loss_cls_weight' : [
            torch.Tensor([1,1,1,1,1])
        ],
    'optimizer' : ['Adam'], 
    'step_size': [1],
    'transforms' : [
            [3, torchvision.transforms.Compose([
                torchvision.transforms.Resize((256,256)),
                torchvision.transforms.ColorJitter(hue=.15, saturation=.15),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])]
            # [4, torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256),
            #     ImageNetPolicy(),
            #     torchvision.transforms.ToTensor(),
            #     torchvision.transforms.Normalize(rgb_mean, rgb_std)
            # ])]
            # [2, torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256),
            #     torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            #     torchvision.transforms.RandomHorizontalFlip(),
            #     torchvision.transforms.RandomPerspective(distortion_scale=0.3),
            #     torchvision.transforms.ToTensor(),
            #     torchvision.transforms.Normalize(rgb_mean, rgb_std)
            # ])]
        ],
    'val_ratio' : [
            [0.025, 0.2, 0.15, 0.15, 0.1]
        ],
    'weight_decay' : [0]
}

