import logging.config
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
import numpy as np
# from RandAugment import RandAugment
from .autoaugment import ImageNetPolicy
logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Tagging Classification')

rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

class SquarePad():
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class Transforms():
    def __init__(self):
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.resolution = (256,256)
        self.cropSize = (224,224)
        self.trainTransform = None
        self.testTransform = None
        self.trainCompose = []
        self.testCompose = []

    def set_resolution(self,x,y):
        self.resolution = (x, y)
        self.cropSize = (int(self.resolution[0] * 0.875), int(self.resolution[1] * 0.875))

    def add_trainCompose(self, item):
        self.trainCompose.insert(0,item)

    def add_testCompose(self,item):
        self.testCompose.insert(0,item)

    def train_transform(self):
        if self.trainTransform == None:
            self.trainCompose += [
                #ImageNetPolicy(),
                SquarePad(),
                transforms.Resize(self.resolution),
                transforms.ColorJitter(hue=.1, saturation=.1),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ]

            self.trainTransform = transforms.Compose(self.trainCompose)
        return self.trainTransform

    def test_transform(self):
        if self.testTransform == None:
            self.testCompose += [
                                 SquarePad(),
                                 transforms.Resize(self.resolution),
                                 transforms.ToTensor(),
                                 transforms.Normalize(self.rgb_mean, self.rgb_std)
                                 ]
            self.testTransform = transforms.Compose(self.testCompose)
        return self.testTransform

    def teacher_test_transform(self):
        if self.testTransform == None:
            self.testCompose += [ 
                                  SquarePad(),
                                  transforms.Resize(self.resolution),
                                  transforms.CenterCrop(self.cropSize),
                                  transforms.ToTensor(),
                                  transforms.Normalize(self.rgb_mean, self.rgb_std)
                                ]
            self.testTransform = transforms.Compose(self.testCompose)

        return self.testTransform