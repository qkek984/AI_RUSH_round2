import logging.config

from torchvision import transforms

logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Tagging Classification')

rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

class Transforms():
    def __init__(self):
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.resolution = (256,256)
        self.cropSize = (int(self.resolution[0]*0.875), int(self.resolution[1]*0.875))
        self.trainTransform = None
        self.testTransform = None
        self.trainCompose = [
            transforms.Resize(self.resolution),
            #transforms.RandomRotation(5, expand=True),
            #transforms.CenterCrop(self.cropSize),
            transforms.ColorJitter(hue=.1, saturation=.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.rgb_mean, self.rgb_std)
        ]
        self.testCompose = [
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(self.rgb_mean, self.rgb_std)
        ]

    def set_resolution(self,x,y):
        self.resolution = (x, y)

    def add_trainCompose(self, item):
        self.trainCompose.insert(0,item)

    def add_testCompose(self,item):
        self.testCompose.insert(0,item)

    def train_transform(self):
        if self.trainTransform == None:
            self.trainTransform = transforms.Compose(self.trainCompose)
        return self.trainTransform

    def test_transform(self):
        if self.testTransform == None:
            self.testTransform = transforms.Compose(self.testCompose)
        return self.testTransform
