import torch
import random 
def get_optimizer(s, params, lr, weight_decay):
    if s == 'Adadelta':
        return torch.optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=weight_decay)
    elif s == 'Adagrad':
        return torch.optim.Adagrad(params, lr=lr, lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0, eps=1e-10)
    elif s == 'Adam':
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif s == 'Adamax':
        return torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif s == 'RMSprop':
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)
    elif s == 'SGD':
        momentum = random.choice([0.0, 0.5, 0.9, 0.99]) 
        print("Momentum for SGD = ", momentum)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=0, weight_decay=weight_decay, nesterov=False)

