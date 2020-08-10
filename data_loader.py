import os

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset

POSSIBLE = [ [0,1,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,0,0,1,1], [1,1,0,0,1], [1,1,1,0,1], [0,0,1,0,1], [0,1,0,0,1]] 
ONEHOT = [[0] * 9] * 9
for i in range(len(ONEHOT)):
    ONEHOT[i][i] = 1 

CLASS = ['착용샷', '설치 후 배치컷', '발색샷', '요리완성', '미분류']
CATEGORY = ['가구/인테리어','패션의류','패션잡화','식품','생활/건강','출산/육아','화장품/미용','스포츠/레저']

CLASS2LABEL = {class_ : label for class_, label in zip(CLASS,[0,1,2,3,4])}
CAT2POS = { cat : pos for cat,pos in zip(CATEGORY, POSSIBLE)}
CAT2ONEH = {cat : oneh for cat,oneh in zip(CATEGORY, ONEHOT)}

class TagImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

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
        
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        tag_name = self.data_frame.iloc[idx]['answer']
        sample['label'] = tag_name
        sample['image_name'] = img_name
        sample['category_possible'] = torch.Tensor(CAT2POS[category])
        sample['category_onehot'] = torch.Tensor(CAT2ONEH[category])
        return sample


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]
        # self.df = pd.read_csv(self.root_dir[:-4]+"label")
        # self.metadata = self.get_metadata()
    '''
    def get_metadata(self):
        meta_dic = {}
        for i, row in self.df.iterrows():
            meta_dic[row['image_name']] = row['category_1']
        
        return meta_dic
    '''
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        #category =  self.metadata[img_name]
        #category_possible = CAT2POS[category]
        #category_onehot = CAT2ONEH[category]
        
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        sample['image_name'] = img_name
        #sample['category_possible'] = torch.Tensor(CAT2POS[category])
        #sample['category_onehot'] = torch.Tensor(CAT2ONEH[category])

        return sample