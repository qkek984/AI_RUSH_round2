import logging.config
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
import numpy as np
# from RandAugment import RandAugment
from .autoaugment import ImageNetPolicy
logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Tagging Classification')

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

        self.rgb_mean_2 = [0.485, 0.456, 0.406]
        self.rgb_std_2 = [0.229, 0.224, 0.225]
        self.rgb_mean = [0.5489, 0.5092, 0.4724]
        self.rgb_std = [0.2131, 0.2149, 0.2209]
        self.resolution = (256,256)
        self.cropSize = (224,224)
        self.trainTransform = None
        self.testTransform = None

        self.trainTransform_2 = None
        self.testTransform_2 = None

        self.trainCompose = []
        self.testCompose = []

        self.trainCompose_2 = []
        self.testCompose_2 = []

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
                SquarePad(),
                transforms.Resize(self.resolution),
                transforms.RandomHorizontalFlip(0.5),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ]

            self.trainTransform = transforms.Compose(self.trainCompose)
        return self.trainTransform

    def train_transform_2(self):
        if self.trainTransform_2 == None:
            self.trainCompose_2 += [
                SquarePad(),
                transforms.Resize(self.resolution),
                transforms.RandomHorizontalFlip(0.5),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean_2, self.rgb_std_2)
            ]

            self.trainTransform_2 = transforms.Compose(self.trainCompose_2)
        return self.trainTransform_2


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

    def test_transform_2(self):
        if self.testTransform_2 == None:
            self.testCompose_2 += [
                SquarePad(),
                transforms.Resize(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean_2, self.rgb_std_2)
            ]
            self.testTransform_2 = transforms.Compose(self.testCompose_2)
        return self.testTransform_2


    def teacher_test_transform(self):
        if self.testTransform == None:
            self.testCompose += [
                SquarePad(),
                transforms.Resize(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ]
            self.testTransform = transforms.Compose(self.testCompose)

        return self.testTransform