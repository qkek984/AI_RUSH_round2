import torch
import torch.nn as nn
from tag.classifier.networks.densenet import DenseNet121
from tag.classifier.networks.resnet50 import ResNet50
from tag.classifier.networks.vgg import VGG16
from tag.classifier.networks.efficientnet import EfficientNet_B3
from torch.nn import functional as F

import xgboost as xgb
import nsml
import pickle
import random 

class Ensemble_model(nn.Module):
    def __init__(self, mode='soft', weight=None, pretrained=None,std=[0,0,0]):
        super(Ensemble_model, self).__init__()
        self.name = "Ensemble_model"
        self.vgg = VGG16(pretrained=False, add_std=std[0])
        self.resnet = ResNet50(pretrained=False, add_std=std[1])
        self.efficientnet = EfficientNet_B3(pretrained=False)
        self.densenet = DenseNet121(pretrained=False, add_std=std[2])

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
        
        if pretrained:
            self.load_finetuned(pretrained)

    def __repr__(self):
        return "Ensemble model with " + self.densenet.name + ", " + self.resnet.name + ", " + self.vgg.name        

    def forward(self, x):
        y1= self.vgg(x.clone())
        
        x_re = F.interpolate(x.clone(),(224,224))
        y3 = self.efficientnet(x_re)
        y4= self.densenet(x.clone())

        y2= self.resnet(x)
        if self.mode == 'soft':
            # _, y1 = torch.max(y1,axis=1)
            # _, y2 = torch.max(y3,axis=1)
            # _, y3 = torch.max(y3,axis=1)
            # print(torch.cat([y1.unsqueeze(1),y2.unsqueeze(1),y3.unsqueeze(1)], axis=1))
            if self.weight == None:
                y = (y1 + y2 + y3 + y4) / 4
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
        torch.save(self.resnet.state_dict(), f'{dirname}/model_{self.resnet.name}')
        torch.save(self.vgg.state_dict(), f'{dirname}/model_{self.vgg.name}')
        torch.save(self.efficientnet.state_dict(), f'{dirname}/model_efficientnet')

    def load(self, dirname):
        print("Loading final weight! ")
        if self.mode == 'xgb':
            print("loaded xgboost")
            self.xgb_classifier = pickle.load(open(f"{dirname}/model_xgboost.dat", "rb"))
        elif self.mode == 'stacked':
            self.stacked_fc.load_state_dict(torch.load(f'{dirname}/model_stacked_fc'))
        # self.densenet.load_state_dict(torch.load(f'{dirname}/model_{self.densenet.name}'))
        self.efficientnet.load_state_dict(torch.load(f'{dirname}/model_efficientnet'))
        self.vgg.load_state_dict(torch.load(f'{dirname}/model_{self.vgg.name}'))
        self.resnet.load_state_dict(torch.load(f'{dirname}/model_{self.resnet.name}'))

    def load_finetuned(self, pretrained):
        print(pretrained)
        bind_model(self.vgg)
        nsml.load(checkpoint=pretrained[0][1], session=pretrained[0][0])

        bind_model(self.resnet)
        nsml.load(checkpoint=pretrained[1][1], session=pretrained[1][0])

        bind_model(self.efficientnet)
        nsml.load(checkpoint=pretrained[2][1], session=pretrained[2][0])

        bind_model(self.densenet)
        nsml.load(checkpoint=pretrained[3][1], session=pretrained[3][0])

        if self.mode == 'xgb':
            for name, param in self.densenet.named_parameters():
                param.requires_grad = False
            for name, param in self.resnet.named_parameters():
                param.requires_grad = False
            for name, param in self.vgg.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.densenet.named_parameters():
                if 'classifier' in name : # and self.weight is None
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in self.resnet.named_parameters():
                if 'fc' in name :
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in self.vgg.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in self.efficientnet.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print("Pretrained weight loaded! ")
        bind_model(self)
        
def bind_model(model):
    def load(dirname, **kwargs):
        try:
            print(f'{dirname}/model_{model.name} finetuned weight loaded!')
            model.load_state_dict(torch.load(f'{dirname}/model_{model.name}'))
        except:
            print(f'{dirname}/model_EfficientNet_b3 finetuned weight loaded!')
            model.load_state_dict(torch.load(f'{dirname}/model_EfficientNet_b3'))
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False


    def save(dirname, **kwargs):
        print(f'Trying to save to {dirname}')
    
        if 'Ensemble' in model.name:
            model.save(dirname)
        else:
            torch.save(model.state_dict(), f'{dirname}/model_{model.name}')

    def infer(test_dir, **kwargs):
        return model.format_test(test_dir)

    nsml.bind(load=load, save=save, infer=infer)
