import torch
from torch.autograd import grad


def gradient(outputs, inputs, grad_outputs = None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(outputs, device = outputs.device)
    
    points_grad = grad(outputs = outputs,
                       inputs = inputs,
                       grad_outputs = grad_outputs,
                       create_graph = True
                       )[0]

    return points_grad