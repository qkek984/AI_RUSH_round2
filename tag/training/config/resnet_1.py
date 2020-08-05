from tag.classifier.datasets.dataset import TagImageDataset
from tag.classifier.experiment.experiment_cls import Classification_experiment
from tag.classifier.networks.resnet50 import ResNet50
from tag.training.config.grid_params import grid
from tag.training.config.best_params import best_param

input_size = (256, 256, 3)
classes = ['착용샷', '설치 후 배치컷', '발색샷', '요리완성', '미분류']
config = {
    'model': Classification_experiment,
    'fit_kwargs': [grid, best_param],
    'experiment_kwargs': {
        'network_fn': ResNet50,
        'network_kwargs': {
            'xgb': False,
            'add_std' : 0   
        },
        'dataset_cls': TagImageDataset,
        'name': "ResNet50",
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}

