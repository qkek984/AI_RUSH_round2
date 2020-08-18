import torch
from torch import nn
import time
import pandas as pd

from configuration.config import logger, Transforms
from data_loader import TagImageInferenceDataset
from torch.utils.data import DataLoader
from scipy.stats import gmean
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from custom_loss import LabelSmoothingLoss

def ensemble_training(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    running_loss = 0.0
    total_loss = 0.0
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0
    cat2correct = 0.0 

    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['image']
        xlabel = data['label']
        category_pos = data['category_possible']
        category_oneh = data['category_onehot']
        category = data['category']
        cat2possible = data['cat2possible']

        cat2possible = cat2possible.to(device)
        category = category.to(device)
        x = x.to(device)
        xlabel = xlabel.to(device)
        category_pos = category_pos.to(device)
        category_oneh = category_oneh.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨

        out = model(x,category_oneh, category)
        logit, pred = out

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
    return total_loss / (i + 1), correct / num_data


def ensemble_evaluate(model, test_loader, device, criterion):
    correct = 0.0
    category_correct = 0.0
    cat2correct = 0.0 
    num_data = 0.0
    total_loss = 0.0

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
            category = data['category']
            cat2possible = data['cat2possible']

            cat2possible = cat2possible.to(device)
            category = category.to(device)
            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            x = x.to(device)
            xlabel = xlabel.to(device)

            out = model(x, category_oneh, category)

            logit, pred = out
            correct += torch.sum(pred == xlabel).item()
            prediction = prediction + pred.detach().cpu().tolist()

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(logit, xlabel)
            elif isinstance(criterion, LabelSmoothingLoss):
                loss = criterion(logit, xlabel, category_pos)

            # category_pred = torch.argmax(logit*category_pos, dim=-1)
            # category_correct += torch.sum(category_pred == xlabel).item()

            cat2pred = torch.argmax(logit*cat2possible, dim=-1)
            cat2correct += torch.sum(cat2pred == xlabel).item()
            cat2_prediction = cat2_prediction + cat2pred.cpu().tolist() 

            num_data += xlabel.size(0)
            total_loss += loss.item()
            label = label + xlabel.tolist()

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

def ensemble_inference(model, test_path: str) -> pd.DataFrame:
    """
    :param model: model
    :param test_path: test path
    :return:
    pandas.DataFrame: columns should be include "image_name" and "y_pred".
    """
    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data',
                                       transform=Transforms().test_transform(), onehot=model.onehot, onehot2=model.model.onehot2)

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
            x = x.to(device)
            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            category = category.to(device)

            logit, pred = model(x,category_oneh, category)

            cat2pred = torch.argmax(logit*cat2possible, dim=-1)
            y_category_pred += cat2pred.type(torch.IntTensor).detach().cpu().tolist()

            filename_list += data['image_name']
            y_pred += pred.type(torch.IntTensor).detach().cpu().tolist()

    # ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_pred})
    ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_category_pred})

    return ret    
