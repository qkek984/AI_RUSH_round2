from typing import Callable, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import nsml
import numpy as np
import itertools 
# from tag.classifier.experiment.utils import Metrics, NSMLReportCallback, evaluate
from tag.classifier.datasets.dataset import TagImageDataset
from tag.classifier.experiment.utils import get_optimizer
import pprint
import time
import random 

PRINTER = pprint.PrettyPrinter()

class Classification_experiment:
    
    def __init__(self, network_fn: Callable, dataset_cls: TagImageDataset, dataset_kwargs, network_kwargs, name="ResNet50",to_log=None, prefix=None):
        self.network_fn = network_fn
        self.xgb = network_kwargs.pop('xgb', False)
        self.network_kwargs = network_kwargs

        self.data = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))
        self.model = network_fn(**kwargs_or_empty_dict(network_kwargs)).cuda()

        self.patience_thresh = 2

        self.name = name
        self.prefix = prefix
        
        if to_log is None:
            to_log = ['loss', 'accuracy']
        self.to_log = to_log + [f'val_{key}' for key in to_log]
        self.best_score = 0
        self.patience = 0


    def fit(self, hyper_param_dict, best_param, load_pretrained=None):

        hyper_param_dict = hyper_param_dict[best_param]   
        sorted_keys = sorted(hyper_param_dict)
        combinations = list(itertools.product(*(hyper_param_dict[key] for key in sorted_keys)))        
        self.data.prepare()
        random.shuffle(combinations)   

        print(f"Trying out {len(combinations)} number of hyperparameter combinations!" )

        for id_, (batch_size, class_ratio, epoch, gamma, lr, cls_w, optim, step, [i, transforms], val_ratio, w_decay) in enumerate(combinations): 
            if id_:
                self.model = self.network_fn(**kwargs_or_empty_dict(self.network_kwargs))
            PRINTER.pprint({key :val for key,val in zip(sorted_keys, [batch_size, class_ratio, epoch, gamma, lr, cls_w, optim, step, [i, transforms], val_ratio, w_decay])})
            
            # initialize 
            self.data.set_transforms(transforms)
            self.data.set_class_ratio(class_ratio)            
            self.model = self.model.cuda()

            print("Trainable Parameters :",[ name for name,param in self.model.named_parameters() if param.requires_grad])
            criterion = torch.nn.CrossEntropyLoss(weight= cls_w.cuda())
            optimizer = get_optimizer(optim,self.model.parameters(), lr=lr, weight_decay=w_decay)
            scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

            # dataloader
            trainloader, validationloader = self.data.train_val_gen(batch_size, val_ratio)
            # std_mean , std_std = self.gray_scale_std(trainloader)

            # train the model by epochs
            score = self.train(epoch[0], trainloader, validationloader, scheduler, optimizer, criterion, id =id_)

            if score < 0.75 and epoch[0]:
                print("Score too low", score)
                print("---train finished---")
                self.patience = 0
                continue

            if 'Ensemble' not in self.name:
                self.unfreeze()
    
            self.train(epoch[1], trainloader, validationloader, scheduler, optimizer, criterion, epoch[0], id= id_)
            print("---train finished---")
            self.patience = 0
            # self.metrics(gen=validationloader)

    def train(self,epochs, trainloader, validationloader,scheduler, optimizer, criterion, prev_epoch=None, id=0):

        batch_size = trainloader.batch_size
        best_score = 0
        epoch_score = 0
        for epoch in range(epochs):
            start = time.time() 
            loss_sum = 0
            val_loss_sum = 0

            ypred = []
            ytrue = []

            for iter_, (xx, yy) in enumerate(trainloader):
                xx = xx.cuda()
                yy = yy.squeeze(1).long().cuda()

                if "Efficient" in self.name:
                    xx = F.interpolate(xx,(224,224))
                
                optimizer.zero_grad()
                outputs = self.model(xx)

                ypred.append(outputs)
                ytrue.append(yy)

                # optimization
                if self.xgb:     
                    pass    
                else:
                    loss = criterion(outputs, yy)
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()

            ypred = torch.cat(ypred, axis=0)
            ytrue = torch.cat(ytrue, axis=0)
            
            if self.xgb:
                ypred = ypred.detach().cpu().numpy()
                ytrue = ytrue.detach().cpu().numpy()

                self.model.xgb_classifier.fit(ypred, ytrue)
                ypred = self.model.xgb_classifier.predict(ypred)

            correct = self.count_correct(ypred, ytrue) 

            val_ypred = []
            val_ytrue = []
            with torch.no_grad():
                for valX, valY in validationloader:
                    valX = valX.cuda()
                    valY = valY.squeeze(1).long().cuda()
                    
                    if "Efficient" in self.name:
                        valX = F.interpolate(valX,(224,224))

                    outputs = self.model(valX)
                    
                    if self.xgb:
                        pass
                    else:
                        val_loss = criterion(outputs, valY)
                        val_loss_sum += val_loss.item()

                    val_ypred.append(outputs)
                    val_ytrue.append(valY)

                val_ypred = torch.cat(val_ypred, axis=0)
                val_ytrue = torch.cat(val_ytrue, axis=0)

                if self.xgb:
                    val_ypred = val_ypred.detach().cpu().numpy()
                    val_ytrue = val_ytrue.detach().cpu().numpy()
                    val_ypred = self.model.xgb_classifier.predict(val_ypred)

                val_correct = self.count_correct(val_ypred, val_ytrue) 
                    
            scheduler.step()
            acc = correct / len(trainloader.dataset)
            val_acc = val_correct / len(validationloader.dataset)

            if prev_epoch:
                epoch += prev_epoch

            # arrange output
            if self.xgb:
                logs = {
                    'train_accuracy': acc,   
                    'val_accuracy': val_acc,
                }
                print("Epoch: %d, accuracy : %1.5f,  val accuracy: %1.5f"%(epoch, acc, val_acc))
            else:            
                print("Epoch: %d, batch loss: %1.5f Loss Sum: %1.5f, accuracy : %1.5f, val loss: %1.5f, val accuracy: %1.5f"% (epoch, loss.item(), loss_sum, acc, val_loss_sum, val_acc))
                logs = {
                    'train_loss': loss_sum,
                    'train_accuracy': acc,   
                    'val_loss' : val_loss_sum,
                    'val_accuracy': val_acc,
                    'loss': loss_sum,
                }

                _ , val_ypred = torch.max(val_ypred,axis=1)
                val_ypred = val_ypred.long().detach().cpu().numpy()
                val_ytrue = val_ytrue.detach().cpu().numpy()

            # log
            end = time.time()
            epoch_score = self.log(epoch, val_ypred, val_ytrue, logs, end-start)
            enter = 1 
            if epoch_score < 0.75:
                return epoch_score

            elif epoch_score >= self.best_score:
                self.patience = 0
                self.best_score = epoch_score
                best_score = epoch_score
                enter = 0

                checkpoint = "best_"+ str(epoch)            
                nsml.save(checkpoint=checkpoint)

            elif epoch_score >= best_score:
                best_score = epoch_score
                self.patience = 0

            else:
                self.patience += 1

            if self.patience > self.patience_thresh:
                print("Early stopping!! ")
                return -1       
            if enter:
                checkpoint = "best_"+ str(epoch)            
                nsml.save(checkpoint=checkpoint)
            print("Weight!  ",self.model.w)

        return epoch_score

    def log(self, epoch, val_pred_class, val_true_class, logs=None, elapsed=0):

        cls_report = classification_report(
            y_true=val_true_class,
            y_pred=val_pred_class,
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )

        # log confusion matrix detail
        for label, res in cls_report.items():
            if isinstance(res, dict):
                to_report = {f'val__{self.name}/{label}/{k}': v for k, v in res.items()}
            else:
                to_report = {f'val__{self.name}/{label}': res}

            nsml.report(step=epoch, **to_report)

            # NOTE the val__ is needed for the nsml plotting.
            for k, v in to_report.items():
                logs[k.replace('val__', 'val/')] = v

        # log loss
        nsml_keys = []
        for k in self.to_log:
            if 'val' in k:
                k_ = k.replace("val_", "val__")
            else:
                k_ = f'train__{k}'
            if self.prefix is not None:
                nsml_keys.append(f'{self.prefix}__{k_}')
            else:
                nsml_keys.append(k_)

        nsml.report(
            step=epoch,
            **{nsml_key: self._to_json_serializable(logs.get(k)) for nsml_key, k
               in zip(nsml_keys, self.to_log)}
        )
        # log final score 
        f1_keys = ['val/'+ self.name + '/' +class_+ '/f1-score' for class_ in self.data.classes[1:]]
        score = np.prod([logs[key] for key in f1_keys ]) ** (1/3)
        
        normalized_confusion = confusion_matrix(val_true_class, val_pred_class, normalize='true')
        score = score * (1-normalized_confusion[0][3])
        nsml.report(step=epoch,
            **{'Score': score}
        )
        # f1_keys += ['val/'+ self.name + '/normal/f1-score']
        print(confusion_matrix(val_true_class, val_pred_class, normalize=None))
        print(normalized_confusion)
        # PRINTER.pprint(cls_report)
        print("\t Final score:", score, [(key_, val_) for key_, val_ in zip(f1_keys,[logs[key] for key in f1_keys])],normalized_confusion[0][3])
        print("\t Time elapsed : %2.5f _________________________________\n"%(elapsed))

        return score

    def count_correct(self, outputs, y):
        if self.xgb:
            return np.sum((outputs == y).astype(int))
        else:
            _, outputs = torch.max(outputs, axis=1)
            return (outputs == y).nonzero().shape[0]

    def format_test(self, test_dir: str) -> pd.DataFrame:
        """

        Args:
            test_dir: Path to the test dataset.

        Returns:
            ret: A dataframe with the columns filename and y_pred. One row is the prediction (y_pred)
                for that file (filename). It is important that this format is used for NSML to be able to evaluate
                the model for the leaderboard.

        """
        dataloader, filenames = self.data.test_gen(test_dir=test_dir, batch_size=256)
        y_pred = self.test(dataloader)
        # y_pred = self.model.predict_generator(gen)
        ret = pd.DataFrame({'filename': filenames, 'y_pred': y_pred})
        return ret

    def test(self, testloader):
        ypreds = []
        with torch.no_grad():
            for xx in testloader:
                xx = xx.cuda()

                if "Efficient" in self.name:
                    xx = F.interpolate(xx,(224,224))

                ypred = self.model(xx)
                ypreds.append(ypred)
            
            ypreds = torch.cat(ypreds,axis=0).detach().cpu().numpy()

        if self.xgb:
            return self.model.xgb_classifier.predict(ypreds)
        else:
            return np.argmax(ypreds, axis=1)

    def gray_scale_std(self, trainloader):
        std_list = []
        for iter_, (xx, yy) in enumerate(trainloader):
            xx = xx.permute(0,2,3,1).numpy() * 256
            xx = xx.astype(np.int)


            for i in range(xx.shape[0]):
                gray_img = np.dot(xx[i][...,:3], [0.2989, 0.5870, 0.1140])
                # np.histogram(img,bin , range) 
                hist,bins = np.histogram(gray_img.ravel(), 256, [0,256])
                mids = 0.5*(bins[1:] + bins[:-1])
                mean = np.average(mids, weights=hist)
                std = np.average((mids - mean)**2, weights=hist)**0.5

                std_list.append(std)

        std_list = np.array(std_list)
        return np.mean(std_list), np.std(std_list)

    def unfreeze(self) -> None:
        for name, params in self.model.named_parameters():
            params.requires_grad = True

    def fit_metrics(self) -> List[str]:
        return ['accuracy']

    def _to_json_serializable(self, v):
        return v if not isinstance(v, np.float32) else v.item()
 
 
    def evaluate(self, data_gen):
        y_pred = []
        y_true = []
        with torch.no_grad():
            for xx, yy in data_gen:
                xx = xx.cuda()
                yy = yy.squeeze(1).long().cuda()

                if "Efficient" in self.name:
                    valX = F.interpolate(valX,(224,224))

                ypred = self.model(xx)
                y_pred.append(ypred)
                y_true.append(yy)

        y_true = torch.cat(y_true,axis=0).detach().cpu().numpy()
        y_pred = torch.cat(y_pred,axis=0).detach().cpu().numpy()
        if self.xgb:
            return self.network_fn.xgb_classifier.predict(y_pred), y_true
        else:
            return np.argmax(ypreds, axis=1), y_true

    def metrics(self, gen) -> None:
        """
        Generate and print metrics.

        Args:
            gen: Keras generator for which to get metrics
            n_batches: How many batches that can be fetched from the data generator.
        """
        y_true, y_pred = self.evaluate(data_gen=gen)

        cls_report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )
        print(
            f'Classification report for validation dataset:\n-----------------------------\n{cls_report}\n=============\n')


def bind_model(experiment: Classification_experiment):
    """
    Utility function to make the model work with leaderboard submission.
    """

    def load(dirname, **kwargs):
        print("Load from model.py")
        if 'Ensemble' in experiment.name:
            experiment.model.load(dirname)
        else:
            experiment.model.load_state_dict(torch.load(f'{dirname}/model_{experiment.name}'))
            experiment.model.eval()
            for name, param in experiment.model.named_parameters():
                param.requires_grad = False


    def save(dirname, **kwargs):
        filename = f'{dirname}/model_{experiment.name}'
        print(f'Trying to save to {filename}')
    
        if 'Ensemble' in experiment.name:
            experiment.model.save(dirname)
        else:
            torch.save(experiment.model.state_dict(), f'{dirname}/model_{experiment.name}')

    def infer(test_dir, **kwargs):
        print("Infer step!")
        return experiment.format_test(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
    