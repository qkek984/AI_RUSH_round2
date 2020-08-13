import logging.config

from torchvision import transforms

logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Tagging Classification')

rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(hue=.1, saturation=.1),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std)
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ColorJitter(hue=.1, saturation=.1),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std)
])

efficientnetb8_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ColorJitter(hue=.1, saturation=.1),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2.0 - 1.0)
])

test_transform = transforms.Compose([
    # transforms.Resize((600, 600)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std)
])

