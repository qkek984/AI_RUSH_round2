import os

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset
from category import *

class TagImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str, onehot2: int , transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.onehot2 = onehot2
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')

        category = self.data_frame.iloc[idx]['category_1']
        category2 = self.data_frame.iloc[idx]['category_2']        
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        tag_name = self.data_frame.iloc[idx]['answer']
        sample['label'] = tag_name
        sample['image_name'] = img_name
        
        sample['category_possible'] = torch.Tensor(CAT2POS[category])
        sample['category_onehot'] = torch.Tensor(CAT2ONEH[category] + (cat22oneh(category,category2) if self.onehot2 else []))
        sample['category'] = torch.Tensor([ CAT2NUM[category] ])
        
        return sample


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]
        self.data_list.remove('test_input') 
        self.df = pd.read_csv(self.root_dir + "/test_input")
        self.metadata = self.get_metadata()
    def get_metadata(self):
        meta_dic = {}
        for i, row in self.df.iterrows():
            meta_dic[row['image_name']] = row['category_1']
        
        return meta_dic
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        category =  self.metadata[img_name]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        sample['image_name'] = img_name
        sample['category_possible'] = torch.Tensor(CAT2POS[category])
        sample['category'] = torch.Tensor([CAT2NUM[category]])
        sample['category_onehot'] = torch.Tensor(CAT2ONEH[category])
        
        return sample