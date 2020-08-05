import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Tuple
from warnings import warn

import pandas as pd
from nsml.constants import DATASET_PATH
from torch.utils import data
import torchvision.transforms.functional as TF
import torchvision
import torch

import numpy as np 
import PIL

UNLABELED = -1 
CLASS2LABEL = {class_ : label for class_, label in zip(['착용샷', '설치 후 배치컷', '발색샷', '요리완성', '미분류'],[0,1,2,3,4])}
RGB_MEAN = [0.485, 0.456, 0.406] 
RGB_STD = [0.229, 0.224, 0.225]

class TagImageDataset(data.Dataset):
    def __init__(self, classes=['착용샷', '설치 후 배치컷', '발색샷', '요리완성', '미분류'], input_size=[256,256,3], partition=None, transforms=None, mode='train', base_dir=None, num_imgs_per_class={}, class_ratio=[0.25,0.25,0.25,0.25], val_split=0.2):

        self.classes = classes
        self.input_size = input_size
        self.base_dir = Path(mkdtemp()) if base_dir is None else base_dir

        self.num_imgs_per_class = num_imgs_per_class
        self.partition = partition

        self.transforms = transforms
        self.mode = mode

        self.val_transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((256,256)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(RGB_MEAN, RGB_STD)
                            ])
        
        self.class_ratio = class_ratio
        self.val_split = val_split
        self._len = None

    def __getitem__(self,idx):
        if self.mode == 'train' or self.mode == 'val':
            idx2 = self.convert_idx(idx)
            dir_ = self.base_dir / 'train' / idx2[1] 
            img_dir = dir_ / os.listdir(dir_)[idx2[0]]
            
            print(np.array(PIL.Image.open(open(img_dir, 'rb'))).shape)
            img = PIL.Image.open(open(img_dir, 'rb')).convert('RGB')
            if self.transforms and self.mode == 'train':
                img = self.transforms(img)
            else:
               img = self.val_transform(img)
            
            return img, torch.Tensor([CLASS2LABEL[idx2[1]]])
        
        elif self.mode == 'test':
            dir_ =  str(self.base_dir)+ "/" + os.listdir(self.base_dir)[idx]
            img = self.val_transform(PIL.Image.open(open(dir_, 'rb')))
            return img

    def convert_idx(self, idx):
        sum_ = 0
        for class_ in self.num_imgs_per_class:
            if idx < sum_ + self.num_imgs_per_class[class_]:
                idx -= sum_
                return [idx, class_]
            sum_ += self.num_imgs_per_class[class_]

    def set_transforms(self, transforms):
        self.transforms = transforms

    def set_class_ratio(self, cls_ratio):
        self.class_ratio = cls_ratio


    def len(self, dataset):
        """
        Utility function to compute the number of datapoints in a given dataset.
        """
        # os.walk returns list of tuples containing list (directiory, [folders], [files])
        if self._len is None:
            self._len = {
                dataset: sum([len(files) for r, d, files in os.walk(self.base_dir / dataset)]) for dataset in ['train']}
            self._len['train'] = int(self._len['train'] * (1 - self.val_split))
            self._len['val'] = int(self._len['train'] * self.val_split)
        return self._len[dataset]

    def prepare(self):
        """
        The resulting folder structure is compatible with the Keras function that generates a dataset from folders.
        """
        dataset = 'train'
        self._initialize_directory(dataset)
        self._rearrange(dataset)

    def _initialize_directory(self, dataset: str) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / dataset
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()

    def _rearrange(self, dataset: str) -> None:
        """
        Then rearranges the files based on the attached metadata. The resulting format is
        --
         |-train
             |-normal
                 |-img0
                 |-img1
                 ...
             |-montone
                 ...
             |-screenshot
                 ...
             |_unknown
                 ...
        """
        output_dir = self.base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        self.num_imgs_per_class = {}

        for class_, label in zip(self.classes, [0,1,2,3,4]):
            self.num_imgs_per_class[class_] = metadata[metadata['answer'] == label].shape[0]

        for _, row in metadata.iterrows():
            src = src_dir / 'train_data' / row['image_name']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / self.classes[row['answer']] / row['image_name']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)

    def train_val_gen(self,batch_size: int, val_ratio : list):
        '''
        Splits the train_data folder into train/val generators. Applies some image augmentation for the train dataset.

        Args:
            batch_size: int

        Returns:
            train_generator: Pytorch dataloader.
            val_generator: Pytorch dataloader.
        '''        
        num_total = self.len('train') + self.len('val') 
        split_num = self.len('train')
        np.random.seed(42)

        sum_ = 0
        val_idx = []
        train_idx = []

        # for i,class_ in enumerate(self.num_imgs_per_class):

        #     prev = sum_
        #     sum_ += self.num_imgs_per_class[class_]

        #     class_val_size = int(self.num_imgs_per_class[class_] * val_ratio[i])
        #     class_train_size = int(num_total * self.class_ratio[i])

        #     add_val_idx = np.random.choice(list(range(prev, sum_)), class_val_size, replace=False)
        #     add_train_idx = list(set(range(prev, sum_)) - set(add_val_idx))

        #     if self.num_imgs_per_class[class_] - class_val_size > class_train_size:
        #         train_idx = train_idx + list(np.random.choice(add_train_idx, class_train_size, replace=False))
        #     else:
        #         repeat = class_train_size // (self.num_imgs_per_class[class_] - class_val_size)
        #         train_idx = train_idx + add_train_idx * repeat 
        #         train_idx = train_idx + list(np.random.choice(train_idx, class_train_size - (self.num_imgs_per_class[class_] - class_val_size) * repeat, replace=False))
            
        #     val_idx = val_idx + list(add_val_idx)

        # for test
        train_idx = np.random.choice(range(num_total), 1000, replace=False)
        val_idx = np.random.choice(range(num_total), 1000, replace=False)

        train_sampler = data.SubsetRandomSampler(train_idx)
        val_sampler = data.SubsetRandomSampler(val_idx)

        partition = {}
        partition['train'] = train_idx
        partition['validation'] = val_idx
        params = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 2}
        # dataloader
        training_set = TagImageDataset(partition=partition['train'],classes=self.classes, input_size=self.input_size,mode='train', transforms=self.transforms,num_imgs_per_class=self.num_imgs_per_class, base_dir=self.base_dir)
        train_loader = data.DataLoader(training_set,sampler=train_sampler, **params)

        validation_set = TagImageDataset(partition=partition['validation'],classes=self.classes, input_size=self.input_size,mode='val', transforms=self.transforms, num_imgs_per_class=self.num_imgs_per_class,base_dir=self.base_dir)
        val_loader = data.DataLoader(validation_set, sampler=val_sampler, **params)

        print("Dataloader constructed! \n\t Train size = %d \tValidation size = %d"%(len(train_idx), len(val_idx)))
        val_composition = [ int(self.num_imgs_per_class[a] * b) for a,b in zip(self.num_imgs_per_class, val_ratio)]
        print("\t Validation dataset composition ", [int(cls_ / sum(val_composition) * 100) for cls_ in val_composition])
        train_composition = [ int(num_total * r) for r in self.class_ratio]
        print("\t Training dataset composition ", [cls_ for cls_ in train_composition])
        print("\t Training dataset proportion ", [int(cls_ / sum(train_composition) * 100) for cls_ in train_composition])

        return train_loader, val_loader

    def test_gen(self, test_dir: str, batch_size: int):
        files = [str(p.name) for p in (Path(test_dir) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]

        test_data = TagImageDataset(partition=list(range(len(files))), base_dir=Path(test_dir) / 'test_data', mode='test')
        test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False,  num_workers=2)

        return test_data_loader, files

    def __len__(self):
        return len(self.partition)