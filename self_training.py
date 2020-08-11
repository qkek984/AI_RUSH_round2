
import os

import nsml
import torch
from nsml import DATASET_PATH

from torch.utils.data import DataLoader
import pandas as pd

import nsml_utils as nu
from configuration.config import logger, test_transform
from data_loader import TagImageDataset
from utils import select_model, get_confidence_score, unclassified_predict
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

def unclassified_df(df, answer=[0,1,2,3,4]):
    columns = [col for col in df]
    unclassifiedData=[]
    for i in range(len(df['answer'])):
        if df['answer'][i] in answer:
            item = []
            for j in range(0, len(columns)):
                item.append(df[columns[j]][i])
            unclassifiedData.append(item)
    unclassified_df = pd.DataFrame(unclassifiedData, columns=columns)

    logger.info(f'unclassifiedData Len: {len(unclassifiedData)} ')
    return unclassified_df

def relabeled_df(df, predictedUnclassified, undersample_ratio= [1, 1, 1, 1, 1], data_cross=False):
    columns = [col for col in df]
    len_columns = len(columns)
    tmpData = [[] for i in range(5)]
    if data_cross:
        undersampledData = [[],[]]
    else:
        undersampledData = [[] for i in range(5)]
    reclassifiedData = []

    len_df_answer =len(df['answer'])
    len_predictedUnclassified = len(predictedUnclassified[0])
    for i in range(len_predictedUnclassified):
        fname = predictedUnclassified[0][i]
        predict = predictedUnclassified[1][i]
        prob = predictedUnclassified[2][i]
        item = (prob,fname,predict)
        tmpData[predict].append(item)

    logger.info(f'undersampling ratio: {undersample_ratio} ')
    for i in range(5):
        if tmpData[i]:
            tmpData[i] = sorted(tmpData[i], reverse=True)
            print("relabeled len: ",len(tmpData[i]), ", [",i,"] score : ",max(tmpData[i])," ~ ", min(tmpData[i]))
            end_idx = int(len(tmpData[i])*undersample_ratio[i])
            tmpData[i] = tmpData[i][:end_idx]
            print("undersampled len: ", len(tmpData[i]), ", [", i, "] score : ", max(tmpData[i]), " ~ ", min(tmpData[i]))
            if data_cross:
                for j in range(len(tmpData[i])):
                    undersampledData[0].append(tmpData[i][j][1]) # fname
                    undersampledData[1].append(tmpData[i][j][2]) # predict
            else:
                for j in range(len(tmpData[i])):
                    undersampledData[i].append(tmpData[i][j][1])
            print("---")

    if data_cross:
        logger.info('data cross ')
        for i in range(len_df_answer):
            if df['image_name'][i] in undersampledData[0]:
                idx = undersampledData[0].index(df['image_name'][i])
                modefy_answer = undersampledData[1][idx]
                item = []
                for j in range(len_columns):
                    item.append(df[columns[j]][i])
                item[4] = modefy_answer
                reclassifiedData.append(item)
            if i%10000 == 0:
                logger.info(f'relabeled {i}/{len_df_answer}')
    else:
        logger.info('disable data cross')
        for i in range(len_df_answer):
            label_idx = df['answer'][i]
            if df['image_name'][i] in undersampledData[label_idx]:
                item = []
                for j in range(len_columns):
                    item.append(df[columns[j]][i])
                reclassifiedData.append(item)
            if i%10000 == 0:
                logger.info(f'relabeled {i}/{len_df_answer}')

    reclassified_df = pd.DataFrame(reclassifiedData, columns=columns)
    return reclassified_df

def df_teacher(teacher_sess_name, undersample_ratio, data_cross, onehot):
    # setting #######################
    batch_size =512
    num_workers = 16
    checkpoint ='best'
    sess_name = teacher_sess_name
    #################################

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    logger.info('Build teacher Model')
    model = select_model('teacher', pretrain=False, n_class=5, onehot=onehot)
    total_param = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model size: {total_param} tensors')
    load_weight(model, 'model.pth')
    model = model.to(device)

    nu.bind_model(model)

    # Set the dataset
    logger.info('Set the dataset')
    df = pd.read_csv(f'{DATASET_PATH}/train/train_label')

    #_, val_df = train_val_df(df, oversample_ratio=[1, 1, 1, 1, 1])# confi-score를 위한 val 데이터 구성
    # testset = TagImageDataset(data_frame=val_df, root_dir=f'{DATASET_PATH}/train/train_data', transform=test_transform)
    # test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    uc_df = unclassified_df(df, answer=[0,1,2,3,4]) #기존 라벨 데이터를 전부 prediction할 unclassified data로 전환
    unclassifiedset = TagImageDataset(data_frame=uc_df, root_dir=f'{DATASET_PATH}/train/train_data', transform=test_transform)
    unclassified_loader = DataLoader(dataset=unclassifiedset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #####get confidence score
    nsml.load(checkpoint, session=sess_name)
    logger.info('[NSML] Model loaded from {}'.format(checkpoint))

    model.eval()
    logger.info('[ST 1] Get confidence score----------')
    #confidence_score = get_confidence_score(model=model, test_loader=test_loader, device=device)
    confidence_score = [0,0,0,0,0]
    #####

    #unclassified class predict
    logger.info('[ST 2] predict Unclassified----------')
    predictedUnclassified = unclassified_predict(model=model, unclassified_loader=unclassified_loader, device=device, confidence_score=confidence_score)
    ##
    logger.info('[ST 3] reclassify----------')
    new_df = relabeled_df(df, predictedUnclassified, undersample_ratio, data_cross)
    logger.info('created reclassified_df !\n')
    return new_df

if __name__ == '__main__':
    new_Df = df_teacher()
