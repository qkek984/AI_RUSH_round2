import os

import nsml
import torch

from utilities.utils import inference
from models.binary_model import Binary_Model
from utilities.binary_utils import binary_inference

def bind_model(model):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model')
        state = torch.load(filename)
        model.load_state_dict(state, strict=True)
        print("Load saved weight from", filename)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model')
        torch.save(model.state_dict(), filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        if isinstance(model,Binary_Model):
            return binary_inference(model, data_path)
        else:
            return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)
