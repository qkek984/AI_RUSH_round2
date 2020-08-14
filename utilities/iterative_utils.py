
import torch
from torch import nn
from scipy.stats import gmean
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def iterative_training(model, train_loader, optimizer, criterion, device, epoch, total_epochs,class_samples):
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

        out = model(x, epoch,category_oneh)
        logit, pred = out

        if isinstance(criterion.loss_fcn, LabelSmoothingLoss):
            loss = criterion(logit, xlabel, pred, category_pos)
        else:
            loss = criterion(logit, xlabel, pred)

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


def iterative_evaluate(model, test_loader, device, criterion, epoch):
    correct = 0.0
    category_correct = 0.0
    num_data = 0.0
    total_loss = 0.0

    label = []
    prediction = []

    with torch.no_grad():
        model.mode = 'evaluate'
        for i, data in enumerate(test_loader):
            x = data['image']
            xlabel = data['label']
            category_pos = data['category_possible']
            category_oneh = data['category_onehot']

            category_pos = category_pos.to(device)
            category_oneh = category_oneh.to(device)
            x = x.to(device)
            xlabel = xlabel.to(device)
            out = model(x, epoch, category_oneh)
        
            logit, pred = out
            if epoch > model.starting_epoch:
                pred = torch.argmax(logit, dim=-1)
                
            if isinstance(criterion.loss_fcn, LabelSmoothingLoss):
                loss = criterion(logit, xlabel, pred, category_pos)
            else:
                loss = criterion(logit, xlabel, pred)
            
            category_pred = torch.argmax(logit*category_pos, dim=-1)
            category_correct += torch.sum(category_pred == xlabel).item()
            correct += torch.sum(pred == xlabel).item()

            num_data += xlabel.size(0)
            total_loss += loss.item()
            label = label + xlabel.tolist()
            prediction = prediction + pred.detach().cpu().tolist()

        del x, xlabel
        model.mode = 'train'

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
