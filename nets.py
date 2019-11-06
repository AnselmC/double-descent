import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_by_name(model_name):
    models = [TwoLayerNN]
    for model in models:
        if model.name() == model_name:
            return model
    
class TwoLayerNN(nn.Module):
    def __init__(self, dim_input, num_hidden, dim_output):
        super(TwoLayerNN, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self.out = nn.Linear(num_hidden, dim_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def name():
        return "two_layer_nn"


def initialize_with_previous_weights(target_weights, previous_weights):
    with torch.no_grad():
        target_weights[:previous_weights.shape[0], :previous_weights.shape[1]] = previous_weights

def initialize_with_previous_bias(target_bias, previous_bias):
    with torch.no_grad():
        target_bias[:len(previous_bias)] = previous_bias
