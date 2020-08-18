import torch
import torch.nn as nn

from models.resnet import ResNet50, resnext50_32x4d, resnet101, resnext101_32x8d
from models.densenet import DenseNet121
from models.utils.load_efficientnet import EfficientNet_B7, EfficientNet_B8, EfficientNet_B5
from models.binary_model import Binary_Model
from utilities.nsml_utils import bind_model
from utilities.utils import inference
from utilities.ensemble_utils import ensemble_inference, ensemble_evaluate
from torch.nn import functional as F

import numpy as np
import xgboost as xgb
import nsml
import pickle
import random 

class Ensemble_Model(nn.Module):
    def __init__(self, args, mode='soft', weight=None, eta=0.3, min_child_weight=1,max_depth=6, gamma=0):
        '''
        XGB parameters:
            https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
            eta = Makes the model more robust by shrinking the weights on each step
            min_child_weight = Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
            max_depth  = Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
            gamma = A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
        '''
        super(Ensemble_Model, self).__init__()
        self.name = "Ensemble_model"
        self.models = [0] * 4
        self.session = [0] * 4
        
        if args.densenet:
            densenet = DenseNet121(pretrained=False)
            args.densenet = args.densenet.split(' ')
            
            if int(args.densenet[1]):
                densenet = Binary_Model(densenet, cat_embed=int(args.densenet[2]), embed_dim=int(args.densenet[3]))
            self.models[0] = densenet
            self.session[0] = args.densenet[0]

        if args.efficientnet_b5:            
            efficientnet = EfficientNet_B5(pretrained=False)
            args.efficientnet_b5 = args.efficientnet_b5.split(' ')
            if int(args.efficientnet_b5[1]):
                efficientnet = Binary_Model(efficientnet, cat_embed=int(args.efficientnet_b5[2]), embed_dim=int(args.efficientnet_b5[3]))
            self.models[1] = efficientnet 
            self.session[1] = args.efficientnet_b5[0]

        if args.efficientnet_b7:            
            efficientnet = EfficientNet_B7(pretrained=False)
            args.efficientnet_b7 = args.efficientnet_b7.split(' ')
            if int(args.efficientnet_b7[1]):
                efficientnet = Binary_Model(efficientnet, cat_embed=int(args.efficientnet_b7[2]), embed_dim=int(efficientnet_b7[3]))
            self.models[1] = efficientnet 
            self.session[1] = args.efficientnet_b7[0]

        if args.resnet:
            resnet = resnext50_32x4d(pretrained=False)
            args.resnet = args.resnet.split(' ')
            if int(args.resnet[1]):
                resnet = Binary_Model(resnet, cat_embed=int(args.resnet[2]), embed_dim=int(args.resnet[3]))
            self.models[2] = resnet
            self.session[2] = args.resnet[0]

        if args.resnet101:
            resnet101 = resnext101_32x8d(pretrained=False)
            args.resnet101 = args.resnet101.split(' ')
            if int(args.resnet101[1]):
                resnet101 = Binary_Model(resnet, cat_embed=int(args.resnet101[2]), embed_dim=int(args.resnet101[3]))
            self.models[3] = resnet101
            self.session[3] = args.resnet101[0]

        self.num_model = len([mod for mod in self.models if mod is not 0])
        self.mode = mode
        self.weight = weight
        if weight is not None:
            # self.weight = [ random.random() for _ in range(4)]
            print("Weight :" ,self.weight)
            print([int(w/ sum(self.weight) * 100) for w in self.weight])
                
        self.w = nn.Parameter(torch.tensor( [ 1 / self.num_model ] * self.num_model).cuda(), requires_grad=True)

        if mode == 'stacked':
            self.stacked_fc = nn.Linear(5 * self.num_model, 4)
        elif mode == 'xgb':
            self.xgb_classifier = xgb.XGBClassifier(objective="multi:softprob", 
                                                    learning_rate=eta, 
                                                    min_child_weight=min_child_weight,
                                                    max_depth=max_depth, 
                                                    gamma=gamma,
                                                    random_state=42)
        
        self.load_finetuned()

    def __repr__(self):
        return "Ensemble model with " + self.densenet.name + ", " + self.resnet.name + ", " + self.vgg.name        

    def forward(self, x, oneh=None, category=None):
        ys = []
        for model in self.models:
            if model:
                if isinstance(model,Binary_Model):
                    y_binary = torch.zeros(x.shape[0], 5).cuda()
                    b_out, class_out, unclass_idx, class_idx = model(x.clone(),oneh,category)
                    y_binary[:,4] = b_out
                    y_binary[:,:4] = class_out
                    ys.append(y_binary)
                else:
                    out, pred = model(x.clone(),oneh)
                    ys.append(out)
                
            
        if self.mode == 'soft':
            # _, y1 = torch.max(y1,axis=1)
            # _, y2 = torch.max(y3,axis=1)
            # _, y3 = torch.max(y3,axis=1)
            # print(torch.cat([y1.unsqueeze(1),y2.unsqueeze(1),y3.unsqueeze(1)], axis=1))
            y = sum(ys)
            return y, torch.argmax(y, dim=-1)

        elif self.mode == "hard":
            ys = [ y_ * w_ for y_, w_ in zip(ys, self.w)]
            y = sum(ys)
            return y, torch.argmax(y, dim=-1)

        elif self.mode == 'stacked':
            ypred = torch.cat(ys, axis=1)
            ypred = self.stacked_fc(ypred)            
            return ypred, torch.argmax(y, dim=-1)

        elif self.mode == "xgb":
            ypred = torch.cat(ys, axis=1)
            return ypred, None

    def save(self, dirname):
        print("Saving final weight! ")
        
        if self.mode == 'xgb':
            pickle.dump(self.xgb_classifier, open(f"{dirname}/model_xgboost.dat", "wb"))
        elif self.mode == 'stacked':
            torch.save(self.stacked_fc.state_dict(), f'{dirname}/model_stacked_fc')    
        # torch.save(self.densenet.state_dict(), f'{dirname}/model_{self.densenet.name}')
        for model in self.models:
            if model:
                torch.save(model.state_dict(), f'{dirname}/model_{model.name}')
        
    def load(self, dirname):
        print("Loading final weight! ")
        if self.mode == 'xgb':
            print("loaded xgboost")
            self.xgb_classifier = pickle.load(open(f"{dirname}/model_xgboost.dat", "rb"))
        elif self.mode == 'stacked':
            self.stacked_fc.load_state_dict(torch.load(f'{dirname}/model_stacked_fc'))
        # self.densenet.load_state_dict(torch.load(f'{dirname}/model_{self.densenet.name}'))
        for model in self.models:
            if model:
                model.load_state_dict(torch.load(f'{dirname}/model_{model.name}'))
 
    def load_finetuned(self):
        for model, sess_ in zip(self.models,self.session):
            print(sess_)
            if model:
                bind_model(model)
                nsml.load(checkpoint='best', session=sess_)
                model = model.cuda()

        if self.mode == 'xgb':
            for model in self.models:
                if model:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
        else:
            for model in self.models:
                if model:
                    for name, param in model.named_parameters():
                        if 'fc' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        print("Pretrained weight loaded! ")
        bind_ensemble_model(self)
        
def bind_ensemble_model(model):
    def load(dirname, **kwargs):
        if model.mode =="xgb":
            model.load(dirname)
        else:
            model.load_state_dict(torch.load(f'{dirname}/model_{model.name}'))

    def save(dirname, **kwargs):
        if model.mode =="xgb":
            model.save(dirname)
        else:
            torch.save(model.state_dict(), f'{dirname}/model_{model.name}')

    def infer(test_dir, **kwargs):
        return ensemble_inference(model, test_dir)

    nsml.bind(load=load, save=save, infer=infer)
