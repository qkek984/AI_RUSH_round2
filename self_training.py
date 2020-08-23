import os

import nsml
import torch
from nsml import DATASET_PATH

from torch.utils.data import DataLoader
import pandas as pd

import utilities.nsml_utils as nu
from configuration.config import *
from models.trainable_embedding import Trainable_Embedding
from models.ensemble_model import Ensemble_Model
from data_loader import TagImageDataset
from utilities.utils import select_model, unclassified_predict
import random


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file).state_dict(), strict=True)
        print('load weight from {}.'.format(weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')

def train_val_df(df, val_ratio = 0.2, n_class = 5, sed=None, oversample_ratio=[1, 1, 1, 1, 1]):
    columns = [col for col in df]
    trainData = [[] for i in range(n_class)]
    valData = [[] for i in range(n_class)]

    # class별로 정리
    for i in range(len(df['answer'])):
        item=[]
        for j in range(0,len(columns)):
            item.append(df[columns[j]][i])
        trainData[df['answer'][i]].append(item)

    # validation 빼놓기
    for i in range(n_class):
        len_td = len(trainData[i])
        val_num = int(len_td * val_ratio)
        if sed:
            random.seed(sed)
        num = [j for j in range(0, len_td)]
        val_num = sorted((random.sample(num, val_num)),reverse=True)
        for vn in val_num:
            valData[i].append(trainData[i].pop(vn))

    # oversampling 구현
    for i in range(n_class):
        if oversample_ratio[i] >= 1:
            trainData[i] = trainData[i] * (oversample_ratio[i] // 1)

            extra = (oversample_ratio[i] % 1) * len(trainData[i])
            trainData[i] += random.sample(trainData[i], extra)
        else:
            trainData[i] = random.sample(trainData[i], int(len(trainData[i])* oversample_ratio[i]))
            valData[i] = random.sample(valData[i], int(len(valData[i]) * oversample_ratio[i]))

    trainSet = []
    valSet = []
    for i in range(n_class):
        trainSet += trainData[i]
        valSet += valData[i]


    print("total trainSet: ", len(trainSet), '\tclass composition : ', [len(l) for l in trainData], [int(len(class_)/sum([len(l) for l in trainData])* 100) for class_ in trainData])
    print("val trainSet: ", len(valSet), '\tclass composition : ', [len(l) for l in valData], [int(len(class_)/sum([len(l) for l in valData])* 100) for class_ in valData])

    train_df = pd.DataFrame(trainSet, columns=columns)
    val_df = pd.DataFrame(valSet, columns=columns)
    return train_df, val_df

def relabeled_df(df, predictedData, undersample_ratio= [1, 1, 1, 1, 1], data_cross=False):
    columns = [col for col in df]
    tmpData={}
    relabeledData = []
    confidence_score = [0, 0, 0, 0, 0]
    len_df_answer = len(df['answer'])

    logger.info(f'undersampling ratio: {undersample_ratio} ')

    for i in range(5):
        if predictedData[i]:
            predictedData[i] = sorted(predictedData[i], reverse=True)
            print("classified len: ",len(predictedData[i]), ", [",i,"] score : ",max(predictedData[i])," ~ ", min(predictedData[i]))
            end_idx = int(len(predictedData[i])*undersample_ratio[i])
            predictedData[i] = predictedData[i][:end_idx]
            print("undersampled len: ", len(predictedData[i]), ", [", i, "] score : ", max(predictedData[i]), " ~ ", min(predictedData[i]))

            for j in range(len(predictedData[i])):
                tmpData[predictedData[i][j][1]] = (predictedData[i][j][2], predictedData[i][j][0])  # tmp[fname]=(predict,prob)
                confidence_score[i] += predictedData[i][j][0]
            confidence_score[i] = confidence_score[i] / len(predictedData[i])
            print("---")

    crossData = [0, 0, 0, 0, 0]
    for i in range(len_df_answer):
        if df['image_name'][i] in tmpData:
            item = list(df.iloc[i])  # item: [cate1, cate2, cate3, cate4, answer, img_name]
            modify_answer = tmpData[df['image_name'][i]][0]  # predict
            if item[4] == modify_answer:
                relabeledData.append(item)
            elif data_cross and tmpData[df['image_name'][i]][1] > confidence_score[modify_answer]:  # prob > conf_prob
                item[4] = modify_answer
                relabeledData.append(item)
                crossData[modify_answer] += 1
            else:
                crossData[modify_answer] += 1
        if i % 10000 == 0:
            logger.info(f'relabeled {i}/{len_df_answer}')
    if data_cross:
        print("Confidence score : ", confidence_score)
        print("Cross data : ", crossData)
    else:
        print("cutted Cross data : ", crossData)
    print("total relabeled data: ",len(relabeledData))

    reclassified_df = pd.DataFrame(relabeledData, columns=columns)
    return reclassified_df


def df_teacher(teacher_sess_name, teacher_model, teacher_cat_embed, undersample_ratio, data_cross, onehot, onehot2, args):
    # setting #######################
    batch_size =256
    num_workers = 16
    checkpoint ='best'
    sess_name = teacher_sess_name
    transform = Transforms()
    
    onehot2= 0

    #################################

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    logger.info('Build teacher Model')
    if teacher_model == "ensemble":
        model = Ensemble_Model(args)
    else:
        model = select_model(teacher_model, pretrain=False, n_class=5, onehot=onehot, onehot2=onehot2)
        if teacher_cat_embed:
            model = Trainable_Embedding(model)
        load_weight(model, 'model.pth')
        nu.bind_model(model)
        nsml.load(checkpoint, session=sess_name)
        logger.info('[NSML] Model loaded from {}'.format(checkpoint))

    model = model.to(device)
    total_param = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model size: {total_param} tensors')
    model.eval()

    # Set the dataset
    logger.info('Set the dataset')
    df = pd.read_csv(f'{DATASET_PATH}/train/train_label')

    df = df.iloc[:1000]

    unclassifiedset = TagImageDataset(data_frame=df, root_dir=f'{DATASET_PATH}/train/train_data', transform=transform.teacher_test_transform(), onehot2=onehot2)
    unclassified_loader = DataLoader(dataset=unclassifiedset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

    #unclassified class predict
    logger.info('[ST 1] predict Unclassifiedset----------')
    predictedData = unclassified_predict(model=model, unclassified_loader=unclassified_loader, device=device)

    logger.info('[ST 2] relabeling----------')
    new_df = relabeled_df(df, predictedData, undersample_ratio, data_cross)
    logger.info('created relabeled_df !\n')
    return new_df

if __name__ == '__main__':
    new_Df = df_teacher()
