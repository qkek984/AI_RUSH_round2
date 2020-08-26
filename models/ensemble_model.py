import torch
import torch.nn as nn

from models.resnet import ResNet50, resnext50_32x4d, resnet101, resnext101_32x8d, resnext101_32x16d
from models.densenet import densenet201, DenseNet121
from models.nest import resnest200
from models.xception import xception
from models.utils.load_efficientnet import EfficientNet_B5,EfficientNet_B6, EfficientNet_B2
from models.binary_model import Binary_Model
from models.trainable_embedding import Trainable_Embedding
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
        self.models = []
        self.session = []
        self.transform = []
        idx = 0        
        if args.densenet:
            args.densenet = args.densenet.split(' ')
            for i in range(len(args.densenet) // 5):
                densenet = densenet201(pretrained=False)
                # densenet = DenseNet121(pretrained=False)
                if int(args.densenet[1 + i*5]):
                    densenet = Binary_Model(densenet, cat_embed=int(args.densenet[2 + i*5]), embed_dim=int(args.densenet[3 + i*5]))
                elif int(args.densenet[2 + i*5]):
                    densenet = Trainable_Embedding(densenet, embed_dim=int(args.densenet[3 + i*5]))
                self.models.append(densenet)
                self.session.append(args.densenet[0 + i*5])
                self.transform.append(int(args.densenet[4 + i*5]))

        if args.nest200:
            args.nest200 = args.nest200.split(' ')
            for i in range(len(args.nest200) // 5):
                nest = resnest200(pretrained=False)

                if int(args.nest200[1 + i * 5]):
                    nest = Binary_Model(nest, cat_embed=int(args.nest200[2 + i * 5]), embed_dim=int(args.nest200[3 + i * 5]))
                elif int(args.nest200[2 + i * 5]):
                    nest = Trainable_Embedding(nest, embed_dim=int(args.nest200[3 + i * 5]))

                self.models.append(nest)
                self.session.append(args.nest200[0 + i * 5])
                self.transform.append(int(args.nest200[4 + i * 5]))

        if args.resnext:
            args.resnext = args.resnext.split(' ')
            for i in range(len(args.resnext) // 5):

                resnet = resnext50_32x4d(pretrained=False)

                if int(args.resnext[1 + i*5]):
                    resnet = Binary_Model(resnet, cat_embed=int(args.resnext[2 + i*5]), embed_dim=int(args.resnext[3 + i*5]))
                elif int(args.resnext[2 + i*5]):
                    resnet = Trainable_Embedding(resnet, embed_dim=int(args.resnext[3 + i*5]))

                self.models.append(resnet)
                self.session.append(args.resnext[0 + i*5])
                self.transform.append(int(args.resnext[4 + i*5]))

        if args.resnext101:
            args.resnext101 = args.resnext101.split(' ')
            for i in range(len(args.resnext101) // 5):
                resnet101 = resnext101_32x8d(pretrained=False)

                if int(args.resnext101[1 + i*5]):
                    resnet101 = Binary_Model(resnet, cat_embed=int(args.resnext101[2 + i*5]), embed_dim=int(args.resnext101[3 + i*5]))
                elif int(args.resnext101[2 + i*5]):
                    resnet101 = Trainable_Embedding(resnet101, embed_dim=int(args.resnext101[3 + i*5]))

                self.models.append(resnet101)
                self.session.append(args.resnext101[0 + i*5])
                self.transform.append(int(args.resnext101[4 + i*5]))

        if args.resnext101_32x16d:
            args.resnext101_32x16d = args.resnext101_32x16d.split(' ')
            for i in range(len(args.resnext101_32x16d) // 5):
                resnet101_32x16d = resnext101_32x16d(pretrained=False)

                if int(args.resnext101_32x16d[1 + i*5]):
                    resnet101_32x16d = Binary_Model(resnet, cat_embed=int(args.resnext101_32x16d[2 + i*5]), embed_dim=int(args.resnext101_32x16d[3 + i*5]))
                elif int(args.resnext101_32x16d[2 + i*5]):
                    resnet101_32x16d = Trainable_Embedding(resnet101_32x16d, embed_dim=int(args.resnext101_32x16d[3 + i*5]))

                self.models.append(resnet101_32x16d)
                self.session.append(args.resnext101_32x16d[0 + i*5])
                self.transform.append(int(args.resnext101_32x16d[4 + i*5]))

        if args.efficient_b2:
            args.efficient_b2 = args.efficient_b2.split(' ')
            for i in range(len(args.efficient_b2) // 5):
                effi = EfficientNet_B2(pretrained=False)

                if int(args.efficient_b2[1 + i * 5]):
                    effi = Binary_Model(resnet, cat_embed=int(args.efficient_b2[2 + i * 5]), embed_dim=int(args.efficient_b2[3 + i * 5]))
                elif int(args.efficient_b2[2 + i * 5]):
                    effi = Trainable_Embedding(effi, embed_dim=int(args.efficient_b2[3 + i * 5]))

                self.models.append(effi)
                self.session.append(args.efficient_b2[0 + i * 5])
                self.transform.append(int(args.efficient_b2[4 + i * 5]))


        if args.efficient_b5:
            args.efficient_b5 = args.efficient_b5.split(' ')
            for i in range(len(args.efficient_b5) // 5):
                effi = EfficientNet_B5(pretrained=False)

                if int(args.efficient_b5[1 + i * 5]):
                    effi = Binary_Model(resnet, cat_embed=int(args.efficient_b5[2 + i * 5]), embed_dim=int(args.efficient_b5[3 + i * 5]))
                elif int(args.efficient_b5[2 + i * 5]):
                    effi = Trainable_Embedding(effi, embed_dim=int(args.efficient_b5[3 + i * 5]))

                self.models.append(effi)
                self.session.append(args.efficient_b5[0 + i * 5])
                self.transform.append(int(args.efficient_b5[4 + i * 5]))

        if args.efficient_b6:
            args.efficient_b6 = args.efficient_b6.split(' ')
            for i in range(len(args.efficient_b6) // 5):
                effi = EfficientNet_B6(pretrained=False)

                if int(args.efficient_b6[1 + i * 5]):
                    effi = Binary_Model(resnet, cat_embed=int(args.efficient_b6[2 + i * 5]), embed_dim=int(args.efficient_b6[3 + i * 5]))
                elif int(args.efficient_b6[2 + i * 5]):
                    effi = Trainable_Embedding(effi, embed_dim=int(args.efficient_b6[3 + i * 5]))

                self.models.append(effi)
                self.session.append(args.efficient_b6[0 + i * 5])
                self.transform.append(int(args.efficient_b6[4 + i * 5]))

        if args.xception:
            args.xception = args.xception.split(' ')
            for i in range(len(args.xception) // 5):
                xcep = xception(pretrained=False)

                if int(args.xception[1 + i*5]):
                    xcep = Binary_Model(resnet, cat_embed=int(args.xception[2 + i*5]), embed_dim=int(args.xception[3 + i*5]))
                elif int(args.xception[2 + i*5]):
                    xcep = Trainable_Embedding(xcep, embed_dim=int(args.xception[3 + i*5]))

                self.models.append(xcep)
                self.session.append(args.xception[0 + i*5])
                self.transform.append(int(args.xception[4 + i*5]))

        self.num_model = len(self.models)
        self.mode = mode
        self.weight = weight
        print("Transforms !" , self.transform)
        if weight is not None:
            print("Weight :" ,self.weight)
            print([int(w/ sum(self.weight) * 100) for w in self.weight])
                
        self.w = nn.Parameter(torch.tensor( [ 1 / self.num_model ] * self.num_model).cuda(), requires_grad=True)

        if mode == 'stacked':
            self.stacked_fc = nn.Linear(5 * self.num_model, 5)

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
        for model, trans in zip(self.models, self.transform):
            if isinstance(model,Binary_Model):
                y_binary = torch.zeros(x.shape[0], 5).cuda()
                b_out, class_out, unclass_idx, class_idx = model(x.clone(),oneh,category)
                y_binary[:,4] = b_out
                y_binary[:,:4] = class_out
                ys.append(y_binary)
            elif isinstance(model,Trainable_Embedding):
                if trans:
                    out, pred = model(x.clone(),oneh,category)
                # else:
                #     out, pred = model(x2.clone(),oneh,category)
                # out = F.softmax(out,dim=-1)
                ys.append(out)
            else:
                if trans:
                    out, pred = model(x.clone(),oneh)
                # else:
                #     out, pred = model(x2.clone(),oneh)
                # out = F.softmax(out,dim=-1)
                ys.append(out)
                
            
        if self.mode == 'soft':
            y = sum(ys)
            return y, torch.argmax(y, dim=-1)

        elif self.mode == "hard":
            ys = [ y_ * w_ for y_, w_ in zip(ys, self.w)]
            y = sum(ys)
            return y, torch.argmax(y, dim=-1)

        elif self.mode == 'stacked':
            ypred = torch.cat(ys, axis=1)
            ypred = self.stacked_fc(ypred)           

            return ypred, torch.argmax(ypred, dim=-1)

        elif self.mode == "xgb":
            ypred = torch.cat(ys, axis=1)
            return ypred, None

    def save(self, dirname):
        print("Saving final weight! ")
        
        if self.mode == 'xgb':
            pickle.dump(self.xgb_classifier, open(f"{dirname}/model_xgboost.dat", "wb"))
        elif self.mode == 'stacked':
            torch.save(self.stacked_fc.state_dict(), f'{dirname}/model_stacked_fc')    
        for i,model in enumerate(self.models):
            if model:
                torch.save(model.state_dict(), f'{dirname}/model_{model.name}_{i}')
        
    def load(self, dirname):
        print("Loading final weight!")
        if self.mode == 'xgb':
            print("loaded xgboost")
            self.xgb_classifier = pickle.load(open(f"{dirname}/model_xgboost.dat", "rb"))
        elif self.mode == 'stacked':
            self.stacked_fc.load_state_dict(torch.load(f'{dirname}/model_stacked_fc'))
        for i, model in enumerate(self.models):
            if model:
                model.load_state_dict(torch.load(f'{dirname}/model_{model.name}_{i}'))
 
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
                for name, param in model.named_parameters():
                    if 'fc' in name and self.mode == 'soft':
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
