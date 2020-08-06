import argparse
import os

import nsml
import torch
from nsml import DATASET_PATH
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import nsml_utils as nu
from configuration.config import logger, train_transform, test_transform
from data_loader import TagImageDataset
from utils import select_optimizer, select_model, evaluate, train
import random

def train_process(args, model, train_loader, test_loader, optimizer, criterion, device):
    best_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        train_loss, train_acc = train(model=model, train_loader=train_loader, optimizer=optimizer,
                                      criterion=criterion, device=device, epoch=epoch, total_epochs=args.num_epoch)
        model.eval()
        test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device, criterion=criterion)

        report_dict = dict()
        report_dict["train__loss"] = train_loss
        report_dict["train__acc"] = train_acc
        report_dict["test__loss"] = test_loss
        report_dict["test__acc"] = test_acc
        report_dict["test__f1"] = test_f1
        report_dict["train__lr"] = optimizer.param_groups[0]['lr']
        nsml.report(False, step=epoch, **report_dict)
        if best_acc < test_acc:
            checkpoint = 'best'
            logger.info(f'[{epoch}] Find the best model! Change the best model.')
            nsml.save(checkpoint)
            best_acc = test_acc
        if (epoch + 1) % 5 == 0:
            checkpoint = f'ckpt_{epoch + 1}'
            nsml.save(checkpoint)

        if (epoch + 1) % args.annealing_period == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            logger.info('Learning rate annealed to : {lr:.6f} @epoch{epoch}'.format(
                epoch=epoch, lr=optimizer.param_groups[0]['lr']))


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

def train_val_df(df, val_ratio = 0.2, class_num = 5):
    columns = [col for col in df]
    trainData = [[] for i in range(0, class_num)]
    valData = [[] for i in range(0, class_num)]

    for i in range(0, len(df['answer'])):
        item=[]
        if df['answer'][i] >= class_num:
            continue

        for j in range(0,len(columns)):
            item.append(df[columns[j]][i])
        trainData[df['answer'][i]].append(item)

    for i in range(0, class_num):
        len_td = len(trainData[i])
        val_num = int(len_td * val_ratio)

        num = [j for j in range(0, len_td)]
        val_num = random.sample(num, val_num)
        minus = 0
        for vn in val_num:
            idx = vn-minus
            valData[i].append(trainData[i].pop(idx))
            minus += 1

    print("trainSet\tvalSet")
    trainSet = []
    valSet = []
    for i in range(0, class_num):
        trainSet += trainData[i]
        valSet += valData[i]
        print(len(trainData[i]),"\t",len(valData[i]))

    trainData = trainData[0]+trainData[1]+trainData[2]+trainData[3]+trainData[4]
    valData = valData[0]+valData[1]+valData[2]+valData[3]+valData[4]
    print("total trainSet: ", len(trainData))
    print("val trainSet: ", len(valData))

    train_df = pd.DataFrame(trainData, columns=columns)
    val_df = pd.DataFrame(valData, columns=columns)
    return train_df, val_df



def main():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Image Tagging Classification from Naver Shopping Reviews')
    parser.add_argument('--sess_name', default='', type=str, help='Session name that is loaded')
    parser.add_argument('--checkpoint', default='best', type=str, help='Checkpoint')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='The number of workers')
    parser.add_argument('--num_epoch', default=100, type=int, help='The number of epochs')
    parser.add_argument('--model_name', default='resnet50', type=str, help='[resnet50, rexnet, dnet1244, dnet1222]')
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning_anneal', default=1.1, type=float)
    parser.add_argument('--annealing_period', default=10, type=int)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--mode', default='train', help='Mode')
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    parser.add_argument('--weight_file', default='model.pth', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    logger.info('Build Model')
    model = select_model(args.model_name, pretrain=args.pretrain, n_class=5)
    total_param = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model size: {total_param} tensors')
    load_weight(model, args.weight_file)
    model = model.to(device)

    nu.bind_model(model)

    if args.pause:
        nsml.paused(scope=locals())

    if args.num_epoch == 0:
        nsml.save('best')
        return

    # Set the dataset
    logger.info('Set the dataset')
    df = pd.read_csv(f'{DATASET_PATH}/train/train_label')
    train_df, val_df = train_val_df(df)
    
    trainset = TagImageDataset(data_frame=train_df, root_dir=f'{DATASET_PATH}/train/train_data',
                               transform=train_transform)
    testset = TagImageDataset(data_frame=val_df, root_dir=f'{DATASET_PATH}/train/train_data',
                              transform=test_transform)

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = select_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay)

    criterion = criterion.to(device)

    if args.mode == 'train':
        logger.info('Start to train!')
        train_process(args=args, model=model, train_loader=train_loader, test_loader=test_loader,
                      optimizer=optimizer, criterion=criterion, device=device)

    elif args.mode == 'test':
        nsml.load(args.checkpoint, session=args.sess_name)
        logger.info('[NSML] Model loaded from {}'.format(args.checkpoint))

        model.eval()
        logger.info('Start to test!')
        test_loss, test_acc, test_f1 = evaluate(model=model, test_loader=test_loader, device=device,
                                                criterion=criterion)
        logger.info(test_loss, test_acc, test_f1)


if __name__ == '__main__':
    main()
