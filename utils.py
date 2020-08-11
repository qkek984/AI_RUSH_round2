import time

import pandas as pd
import torch
from adamp import AdamP, SGDP
from scipy.stats import gmean
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch import optim
from torch.utils.data import DataLoader

from configuration.config import logger, test_transform
from data_loader import TagImageInferenceDataset
from models.teacher_model import Resnet50_FC2
from models.baseline_resnet import Resnet50_FC2
from models.resnet import ResNet50, resnext50_32x4d
from models.densenet import DenseNet121
from models.utils.load_efficientnet import EfficientNet_B7, EfficientNet_B8
from custom_loss import LabelSmoothingLoss

import os

def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    running_loss = 0.0
    total_loss = 0.0
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0

    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['image']
        xlabel = data['label']
        category_pos = data['category_possible']
        category_oneh = data['category_onehot']

        x = x.to(device)
        xlabel = xlabel.to(device)
        category_pos = category_pos.to(device)
        category_oneh = category_oneh.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨

        out = model(x,category_oneh)

        logit, pred = out
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(logit, xlabel)
        elif isinstance(criterion, LabelSmoothingLoss):
            loss = criterion(logit, xlabel, category_pos)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

        category_pred = torch.argmax(logit*category_pos, dim=-1)
        category_correct += torch.sum(category_pred == xlabel).item()

        correct += torch.sum(pred == xlabel).item()
        num_data += xlabel.size(0)
        if i % 100 == 0:  # print every 100 mini-batches
            logger.info("epoch: {}/{} | step: {}/{} | loss: {:.4f} | time: {:.4f} sec".format(epoch, total_epochs, i,
                                                                                              len(train_loader),
                                                                                              running_loss / 2000,
                                                                                              time.time() - start))
            running_loss = 0.0

    logger.info(
        '[{}/{}]\tloss: {:.4f}\tacc: {:.4f} \tcategory_acc : {:.4f}'.format(epoch, total_epochs, total_loss / (i + 1), correct / num_data, category_correct / num_data))
    del x, xlabel
    torch.cuda.empty_cache()
    return total_loss / (i + 1), correct / num_data

def get_confidence_score(model, test_loader, device, defaltThresh=0, n_class=5):
    errorProb = [[] for _ in range(n_class)]
    confid_score = [0 for _ in range(n_class)]
    avg_score = [0 for _ in range(n_class)]
    lentl = len(test_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img_name = data['image_name']
            x = data['image']
            img_label = data['label']
            category_pos = data['category_possible']
            category_oneh = data['category_onehot']

            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            x = x.to(device)

            out = model(x, category_oneh)
            logit, pred = out

            category_pred = torch.argmax(logit * category_pos, dim=-1)

            for item in zip(img_name, img_label, category_pred, logit):
                fname = item[0]
                label = int(item[1])
                predict = int(item[2])
                prob = float(item[3][predict])
                if label != predict:
                    errorProb[predict].append(prob)

            if i % 50 == 0:#작업 경과 출력
                logger.info(f'confidence score {i} / {lentl}')
        del x, img_label, img_name

    #weight=[0.1, 0.25, 0.25, 0.25, 0.01]

    for i in range(n_class):
        ep = sorted(errorProb[i],reverse=True)
        if ep:
            #confid_score[i] = max(ep) # max
            #confid_score[i] = sum(ep) / len(ep) # avg
            confid_score[i] = sum(ep[:10]) / len(ep[:10])  # avg
            #confid_score[i] = ep[int(len(ep) * 0.25)] # 0.25
            #confid_score[i] = ep[int(len(ep) * weight[i])] # weight
        else:
            confid_score[i] = defaltThresh
        logger.info(f'Top 5 error score [{i}] label: {ep[:5]}')
    logger.info(f'condidence score: {confid_score}')
    return confid_score

def unclassified_predict(model, unclassified_loader, device, confidence_score, n_class=5):
    predictedUnclassified = [[],[],[]]
    lenul = len(unclassified_loader)
    with torch.no_grad():
        for i, data in enumerate(unclassified_loader):
            img_name = data['image_name']
            x = data['image']
            category_oneh = data['category_onehot']
            category_oneh = category_oneh.to(device)
            x = x.to(device)

            out = model(x, category_oneh)
            logit, pred = out

            for item in zip(img_name, pred, logit):
                fname = item[0]
                predict = int(item[1])
                prob = float(item[2][predict])

                if prob > confidence_score[predict]:
                    predictedUnclassified[0].append(fname)
                    predictedUnclassified[1].append(predict)
                    predictedUnclassified[2].append(prob)

            if i % 100 == 0:#작업 경과 출력
                logger.info(f'predict unclassied data {i} / {lenul}')
    return predictedUnclassified

def evaluate(model, test_loader, device, criterion):
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0
    total_loss = 0.0

    label = []
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            xlabel = data['label']
            category_pos = data['category_possible']
            category_oneh = data['category_onehot']

            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            x = x.to(device)
            xlabel = xlabel.to(device)

            out = model(x, category_oneh)

            logit, pred = out
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(logit, xlabel)
            elif isinstance(criterion, LabelSmoothingLoss):
                loss = criterion(logit, xlabel, category_pos)

            category_pred = torch.argmax(logit*category_pos, dim=-1)
            category_correct += torch.sum(category_pred == xlabel).item()
            correct += torch.sum(pred == xlabel).item()

            num_data += xlabel.size(0)
            total_loss += loss.item()
            label = label + xlabel.tolist()
            prediction = prediction + pred.detach().cpu().tolist()
        del x, xlabel

    torch.cuda.empty_cache()

    confusion = confusion_matrix(label,prediction)
    confusion_norm = confusion_matrix(label,prediction, normalize='true')
    logger.info(f'\n{confusion}')
    logger.info(f'\n{confusion_norm}')
    
    f1_array = f1_score(label, prediction, average=None)
    
    logger.info(f"f1 score : {f1_array}")
    f1_mean = gmean(f1_array)
    logger.info('validation loss: {loss:.4f}\v validation acc: {acc:.4f} \t validation category acc: {cat_acc:.4f}\v validation F1: {f1:.4f}'
                .format(loss=total_loss / (i + 1), acc=correct / num_data, f1=f1_mean, cat_acc=category_correct / num_data))
    return total_loss / (i + 1), correct / num_data, f1_mean


def inference(model, test_path: str) -> pd.DataFrame:
    """
    :param model: model
    :param test_path: test path
    :return:
    pandas.DataFrame: columns should be include "image_name" and "y_pred".
    """
    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data',
                                       transform=test_transform)

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    y_cat_pred = []
    filename_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            # category_pos = data['category_possible']
            # category_oneh = data['category_onehot']

            x = x.to(device)
            # category_pos = category_pos.to(device)
            # category_oneh = category_oneh.to(device)

            logit, pred = model(x) # , category_oneh

            filename_list += data['image_name']
            # These predictions are yet to be used            
            # category_pred = torch.argmax(logit * category_pos, dim=-1)
            # y_category_pred += category_pred.type(torch.IntTensor).detach().cpu().tolist()

            y_pred += pred.type(torch.IntTensor).detach().cpu().tolist()

    ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_pred})
    # ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_category_pred})

    return ret

def select_model(model_name: str, pretrain: bool, n_class: int, onehot : int):
    if model_name == 'resnet50':
        model = ResNet50(onehot=onehot)
    elif model_name == "resnext":
        model = resnext50_32x4d(onehot=onehot)
    elif model_name == 'teacher':
        model = resnext50_32x4d(onehot=onehot)
    elif model_name == 'densenet':
        model = DenseNet121()
    elif model_name == "efficientnet_b7":
        model = EfficientNet_B7()
    elif model_name == "efficientnet_b8":
        model = EfficientNet_B8()        
    else:
        raise NotImplementedError('Please select in [resnet50, densenet, efficientnet_b7, efficientnet_b8]')
    return model


def select_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = SGDP(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'SGDP':
        optimizer = SGDP(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'Adam':
        return torch.optim.Adam(param, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opt_name == 'AdamP':
        #optimizer = AdamP(param, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, nesterov=True)
        optimizer = AdamP(param, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    else:
        raise NotImplementedError('The optimizer should be in [SGD]')
    return optimizer
