import torch
import torch.nn as nn

from models.resnet import ResNet50, resnext50_32x4d, resnet101, resnext101_32x8d
from models.densenet import DenseNet121
from models.utils.load_efficientnet import EfficientNet_B7, EfficientNet_B8, EfficientNet_B5
from models.binary_model import Binary_Model
from utilities.nsml_utils import bind_model
from utilities.utils import inference
from torch.nn import functional as F

import xgboost as xgb
import nsml
import pickle
import random 

class Ensemble_Model(nn.Module):
    def __init__(self, args, mode='soft', weight=None):
        super(Ensemble_Model, self).__init__()
        self.name = "Ensemble_model"
        self.models = [0] * 3
        self.session = [0] * 3
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
            if int(args.efficientnet[1]):
                efficientnet = Binary_Model(efficientnet, cat_embed=int(args.efficientnet_b5[2]), embed_dim=int(args.efficientnet_b5[3]))
            self.models[1] = efficientnet 
            self.session[1] = args.efficientnet_b5[0]

        if args.efficientnet_b7:            
            efficientnet = EfficientNet_B7(pretrained=False)
            args.efficientnet_b7 = args.efficientnet_b7.split(' ')
            if int(args.efficientnet[1]):
                efficientnet = Binary_Model(efficientnet, cat_embed=int(args.efficientnet_b7[2]), embed_dim=int(efficientnet_b7[3]))
            self.models[1] = efficientnet 
            self.session[1] = args.efficientnet_b7[0]

        if args.resnet:
            resnet = resnext50_32x4d(pretrained=False)
            args.resnet = args.resnet.split(' ')
            if int(args.resnet[1]):
                self.resnet = Binary_Model(resnet, cat_embed=int(args.resnet[2]), embed_dim=int(args.resnet[3]))
            self.models[2] = resnet
            self.session[2] = args.resnet[0]

        self.mode = mode
        self.weight = weight
        if weight is not None:
            # self.weight = [ random.random() for _ in range(4)]
            print("Weight :" ,self.weight)
            print([int(w/ sum(self.weight) * 100) for w in self.weight])
                
        self.w = nn.Parameter(torch.tensor([0.25]*4).cuda(), requires_grad=True)

        if mode == 'stacked':
            self.stacked_fc = nn.Linear(4 * 4, 4)
        elif mode == 'xgb':
            self.xgb_classifier = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
        

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
            if self.weight == None:
                y = sum(ys)
            else:
                # w1, w2, w3, w4 = self.w
                w1, w2, w3, w4 = self.weight
                y = (y1 * w1 + y2* w2 + y3* w3 + y4 * w4)/ sum(self.weight)

            return y

        elif self.mode == 'stacked':
            ypred = torch.cat([y1, y2, y3, y4], axis=1)
            ypred = self.stacked_fc(ypred)            
            return ypred 

        elif self.mode == "xgb":
            ypred = torch.cat([y1, y2, y3,y4], axis=1)

            return ypred

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
        model.load_state_dict(torch.load(f'{dirname}/model_{model.name}'))

    def save(dirname, **kwargs):
        torch.save(model.state_dict(), f'{dirname}/model_{model.name}')

    def infer(test_dir, **kwargs):
        return inference(model, test_dir)

    nsml.bind(load=load, save=save, infer=infer)
