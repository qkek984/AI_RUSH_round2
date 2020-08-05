from importlib import import_module

import nsml

from tag.classifier.experiment.experiment_cls import bind_model


def train(experiment_name: str = 'resnet_1', best_param : int = 0, pause: bool = False, mode: str = 'train'):
    config = import_module(f'tag.training.config.{experiment_name}').config
    model = config['model'](**config['experiment_kwargs'])
    bind_model(model)
    if pause:
        print("before pause")
        nsml.paused(scope=locals())
    if mode == 'train':
        print("before fit")
        model.fit(config['fit_kwargs'], best_param)
