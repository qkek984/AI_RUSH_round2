import torch
from torch import nn
import time
import pandas as pd

from configuration.config import logger, Transforms
from data_loader import TagImageInferenceDataset
from torch.utils.data import DataLoader
from scipy.stats import gmean
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def binary_train(model, train_loader, optimizer, device, epoch, total_epochs):
    running_loss = 0.0
    total_loss = 0.0
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0

    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['image']
        xlabel = data['label']
        pred = torch.zeros(xlabel.shape[0]).long().to(device)

        category_pos = data['category_possible']
        category_oneh = data['category_onehot']
        category = data['category']

        x = x.to(device)
        xlabel = xlabel.to(device)
        category = category.to(device)
        category_pos = category_pos.to(device)
        category_oneh = category_oneh.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨

        out = model(x,category_oneh, category if model.cat_embed else None)
        b_out, class_out, unclass_idx, class_idx = out
        
        if class_idx.shape[0] > 0:
            pred[class_idx] = torch.argmax(class_out[class_idx], dim=-1)
        pred[unclass_idx] = 4

        binary_label = (xlabel[unclass_idx] == 4).float()
        class_label = xlabel[class_idx]
        falpos_idx = (class_label == 4).nonzero().squeeze(1)
        trupos_idx = (class_label < 4).nonzero().squeeze(1)

        binary_label = torch.cat([binary_label, torch.ones(falpos_idx.shape[0]).to(device)])
        b_out = torch.cat([b_out[unclass_idx], b_out[falpos_idx]])

        class_out = class_out[trupos_idx]
        class_label = class_label[trupos_idx]

        bin_loss = model.criterion_1(b_out, binary_label)
        class_loss = model.criterion_2(class_out, class_label) 
        loss = bin_loss + class_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

        # category_pred = torch.argmax(logit*category_pos, dim=-1)
        # category_correct += torch.sum(category_pred == xlabel).item()

        correct += torch.sum(pred == xlabel).item()
        num_data += xlabel.size(0)

        if i % 100 == 0:  # print every 100 mini-batches
            logger.info("epoch: {}/{} | step: {}/{} | loss: {:.4f} | time: {:.4f} sec \t binary_loss {:.4f} \t class_loss {:.4f}".format(epoch+1, total_epochs, i,
                                                                                              len(train_loader),
                                                                                              running_loss / 2000,
                                                                                              time.time() - start,
                                                                                              bin_loss,
                                                                                              class_loss))
            running_loss = 0.0

    logger.info(
        '[{}/{}]\tloss: {:.4f}\tacc: {:.4f}'.format(epoch+1, total_epochs, total_loss / (i + 1), correct / num_data))
    del x, xlabel
    torch.cuda.empty_cache()
    return total_loss / (i + 1), correct / num_data


def binary_evaluate(model, test_loader, device):
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
            pred = torch.zeros(xlabel.shape[0]).long().to(device)

            category_pos = data['category_possible']
            category_oneh = data['category_onehot']
            category = data['category']

            category = category.to(device)
            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            x = x.to(device)
            xlabel = xlabel.to(device)

            out = model(x,category_oneh, category if model.cat_embed else None)
            b_out, class_out, unclass_idx, class_idx = out
            
            if class_idx.shape[0] > 0:
                pred[class_idx] = torch.argmax(class_out[class_idx], dim=-1)
            pred[unclass_idx] = 4

            binary_label = (xlabel[unclass_idx] == 4).float()
            class_label = xlabel[class_idx]
            falpos_idx = (class_label == 4).nonzero().squeeze(1)
            trupos_idx = (class_label < 4).nonzero().squeeze(1)

            binary_label = torch.cat([binary_label, torch.ones(falpos_idx.shape[0]).to(device)])
            b_out = torch.cat([b_out[unclass_idx], b_out[falpos_idx]])

            class_out = class_out[trupos_idx]
            class_label = class_label[trupos_idx]

            bin_loss = model.criterion_1(b_out, binary_label)
            class_loss = model.criterion_2(class_out, class_label) 
            loss = bin_loss + class_loss

            # category_pred = torch.argmax(logit*category_pos, dim=-1)
            # category_correct += torch.sum(category_pred == xlabel).item()
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
    logger.info('validation loss: {loss:.4f}\v validation acc: {acc:.4f} \v validation F1: {f1:.4f}'
                .format(loss=total_loss / (i + 1), acc=correct / num_data, f1=f1_mean))
    return total_loss / (i + 1), correct / num_data, f1_mean

def binary_inference(model, test_path: str) -> pd.DataFrame:
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
            pred = torch.zeros(x.shape[0]).long().cuda()

            category_pos = data['category_possible']
            category_oneh = data['category_onehot']
            category = data['category']

            x = x.to(device)
            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            category = category.to(device)

            out = model(x,category_oneh, category if model.cat_embed else None)
            sig_b_out, class_out, unclass_idx, class_idx = out

            pred[class_idx] = torch.argmax(class_out, dim=-1)
            pred[unclass_idx] = 4

            filename_list += data['image_name']
            # These predictions are yet to be used
            y_pred += pred.type(torch.IntTensor).detach().cpu().tolist()

    ret = pd.DataFrame({'image_name': filename_list, 'y_pred': y_pred})

    return ret    