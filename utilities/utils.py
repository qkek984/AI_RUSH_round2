import time

import pandas as pd
import torch
from adamp import AdamP, SGDP
from scipy.stats import gmean
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch import optim
from torch.utils.data import DataLoader

from configuration.config import logger, Transforms
from data_loader import TagImageInferenceDataset
from models.teacher_model import Resnet50_FC2
from models.baseline_resnet import Resnet50_FC2
from models.resnet import ResNet50, resnext50_32x4d, resnet101, resnext101_32x8d, resnext101_32x16d
from models.densenet import DenseNet121
from models.utils.load_efficientnet import EfficientNet_B7, EfficientNet_B8, EfficientNet_B5
from custom_loss import LabelSmoothingLoss

import os

def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    running_loss = 0.0
    total_loss = 0.0
    correct = 0.0
    category_correct = 0.0
    cat2correct = 0.0 
    num_data = 0.0
    # stds = []
    # means = []
    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['image']
        xlabel = data['label']
        category_pos = data['category_possible']
        category_oneh = data['category_onehot']
        cat2possible = data['cat2possible']

        cat2possible = cat2possible.to(device)
        x = x.to(device)
        xlabel = xlabel.to(device)
        category_pos = category_pos.to(device)
        category_oneh = category_oneh.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨

        out = model(x,category_oneh)
        logit, pred = out

        # x = x.reshape(x.shape[0], x.shape[1], -1)
        # stds.append(torch.std(x, axis=2))
        # means.append(torch.mean(x, axis=2))

        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(logit, xlabel)
        elif isinstance(criterion, LabelSmoothingLoss):
            loss = criterion(logit, xlabel, category_pos)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

        # category_pred = torch.argmax(logit*category_pos, dim=-1)
        # category_correct += torch.sum(category_pred == xlabel).item()

        cat2pred = torch.argmax(logit*cat2possible, dim=-1)
        cat2correct += torch.sum(cat2pred == xlabel).item()

        correct += torch.sum(pred == xlabel).item()
        num_data += xlabel.size(0)
        if i % 100 == 0:  # print every 100 mini-batches
            logger.info("epoch: {}/{} | step: {}/{} | loss: {:.4f} | time: {:.4f} sec".format(epoch+1, total_epochs, i,
                                                                                              len(train_loader),
                                                                                              running_loss / 2000,
                                                                                              time.time() - start))
            running_loss = 0.0

    logger.info(
        '[{}/{}]\tloss: {:.4f}\tacc: {:.4f} \tcategory_acc : {:.4f}'.format(epoch+1, total_epochs, total_loss / (i + 1), correct / num_data, cat2correct / num_data))
    del x, xlabel
    torch.cuda.empty_cache()
    # means = torch.cat(means, axis=0)
    # stds = torch.cat(stds, axis=0)

    return total_loss / (i + 1), correct / num_data

def unclassified_predict(model, unclassified_loader, device, n_class=5):
    predictedData = [[] for i in range(n_class)]
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
                predict= int(item[1])
                predictedData[predict].append((float(item[2][predict]), item[0], predict))# prob, fname, predict

            if i % 100 == 0:#작업 경과 출력
                logger.info(f'predict unclassied data {i} / {lenul}')
    return predictedData

def evaluate(model, test_loader, device, criterion):
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0
    total_loss = 0.0
    cat2correct = 0.0
    label = []
    prediction = []
    cat_prediction = []
    cat2_prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            xlabel = data['label']
            category_pos = data['category_possible']
            category_oneh = data['category_onehot']
            cat2possible = data['cat2possible']

            cat2possible = cat2possible.to(device)
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

            correct += torch.sum(pred == xlabel).item()

            # category_pred = torch.argmax(logit*category_pos, dim=-1)
            # category_correct += torch.sum(category_pred == xlabel).item()

            cat2pred = torch.argmax(logit*cat2possible, dim=-1)
            cat2correct += torch.sum(cat2pred == xlabel).item()

            num_data += xlabel.size(0)
            total_loss += loss.item()
            label = label + xlabel.tolist()

            prediction = prediction + pred.detach().cpu().tolist()
            # cat_prediction = cat_prediction + category_pred.cpu().tolist()
            cat2_prediction = cat2_prediction + cat2pred.cpu().tolist() 
        del x, xlabel

    torch.cuda.empty_cache()

    confusion = confusion_matrix(label,cat2_prediction)
    confusion_norm = confusion_matrix(label,cat2_prediction, normalize='true')
    logger.info(f'\n{confusion}')
    logger.info(f'\n{confusion_norm}')
    
    f1_array = f1_score(label, cat2_prediction, average=None)
    
    logger.info(f"f1 score : {f1_array}")
    f1_mean = gmean(f1_array)
    logger.info('validation loss: {loss:.4f}\v validation acc: {acc:.4f} \t validation category acc: {cat_acc:.4f}\v validation F1: {f1:.4f}'
                .format(loss=total_loss / (i + 1), acc=correct / num_data, f1=f1_mean, cat_acc=cat2correct / num_data))
    return total_loss / (i + 1), correct / num_data, f1_mean


def inference(model, test_path: str) -> pd.DataFrame:
    """
    :param model: model
    :param test_path: test path
    :return:
    pandas.DataFrame: columns should be include "image_name" and "y_pred".
    """
    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data',
                                       transform=Transforms().test_transform(), onehot=model.onehot, onehot2=model.onehot2)

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    y_cat_pred = []
    y_category_pred=[]
    filename_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            category_pos = data['category_possible']
            category_oneh = data['category_onehot']
            category = data['category']
            cat2possible = data['cat2possible']

            cat2possible = cat2possible.to(device)
            category = category.to(device)
            x = x.to(device)
            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)

            logit, pred = model(x, category_oneh) # ,
            #y_pred += pred.type(torch.IntTensor).detach().cpu().tolist()

            filename_list += data['image_name']
            # category_pred = torch.argmax(logit * category_pos, dim=-1)
            # y_category_pred += category_pred.type(torch.IntTensor).detach().cpu().tolist()

            cat2pred = torch.argmax(logit*cat2possible, dim=-1)
            y_category_pred += cat2pred.type(torch.IntTensor).detach().cpu().tolist()

    # ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_pred})
    ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_category_pred})

    return ret

def select_model(model_name: str, pretrain: bool, n_class: int, onehot : int, onehot2=0):
    if model_name == 'resnet50':
        model = ResNet50(onehot=onehot,onehot2=onehot2)
    elif model_name == "resnext":
        model = resnext50_32x4d(onehot=onehot,onehot2=onehot2)
    elif model_name == "resnet101":
        model = resnet101(onehot=onehot,onehot2=onehot2)
    elif model_name == "resnext101":
        model = resnext101_32x8d(onehot=onehot,onehot2=onehot2)
    elif model_name == "resnet101_32x16d":
        model = resnext101_32x16d(onehot=onehot, onehot2=onehot2)
    elif model_name == 'densenet':
        model = DenseNet121(onehot=onehot,onehot2=onehot2)
    elif model_name == "efficientnet_b5":
        model = EfficientNet_B5(onehot=onehot,onehot2=onehot2)
    elif model_name == "efficientnet_b7":
        model = EfficientNet_B7(onehot=onehot,onehot2=onehot2)
    elif model_name == "efficientnet_b8":
        model = EfficientNet_B8(onehot=onehot,onehot2=onehot2)        
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
        optimizer = AdamP(param, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError('The optimizer should be in [SGD]')
    return optimizer
