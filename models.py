import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.optim.lr_scheduler import MultiStepLR


def get_model_by_name(model_name):
    models = [TwoLayerNN, RandomFourier, RandomFourierReLU]
    for model in models:
        if model.name() == model_name:
            return model


class RandomFourierReLU(nn.Module):
    pass

# one-vs-all


class RandomFourier(nn.Module):
    def __init__(self, dim_input, num_hidden):
        super(RandomFourier, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self.hidden.no_grad()  # fix weights
        self.out = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = exp(self.hidden(x))
        x = self.out(x)
        return x

    def train_step():
        pass

    def val_step():
        pass

    def test_step():
        pass

    @staticmethod
    def name():
        return "random_fourier"


class TwoLayerNN(nn.Module):
    def __init__(self, dim_input, num_hidden, dim_output):
        super(TwoLayerNN, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self.out = nn.Linear(num_hidden, dim_output)
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer = optimizers.SGD(
            self.parameters(), lr=0.001, momentum=0.95)
        self.scheduler = MultiStepLR(self._optimizer, milestones=list(
            range(500, 6000, 500)), gamma=0.9)
        self._train_step = 0
        self._val_step = 0
        self._test_step = 0

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self, prev_model):
        if prev_model is None:
            # initialize smallest net using Xavier Glorot-uniform distribution
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(self.hidden.weight, gain=gain)
            nn.init.xavier_uniform_(self.out.weight, gain=gain)
        else:
            nn.init.normal_(self.hidden.weight, 0, 0.01)
            initialize_with_previous_weights(
                self.hidden.weight, prev_model.hidden.weight)
            initialize_with_previous_bias(
                self.hidden.bias, prev_model.hidden.bias)
            nn.init.normal_(self.out.weight, 0, 0.01)

    def epoch_step(self):
        self._train_step = 0
        self._val_step = 0

    def train_step(self, data, device):
        self._train_step += 1
        if self._train_step == 1:
            self.train()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        self._optimizer.zero_grad()
        outputs = self(inputs)
        loss = self._loss_function(outputs, labels)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def val_step(self, data, device):
        self._val_step += 1
        if self._val_step == 1:
            self.eval()
        images, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(images)
        loss = self._loss_function(outputs, labels)
        return loss.item()

    def test_step(self, data, device):
        self._test_step += 1
        if self._test_step == 1:
            self.eval()
        images, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(images)
        loss = self._loss_function(outputs, labels)
        test_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return test_loss, total, correct
            
        pass

    @staticmethod
    def name():
        return "two_layer_nn"


def initialize_with_previous_weights(target_weights, previous_weights):
    with torch.no_grad():
        target_weights[:previous_weights.shape[0],
                       :previous_weights.shape[1]] = previous_weights


def initialize_with_previous_bias(target_bias, previous_bias):
    with torch.no_grad():
        target_bias[:len(previous_bias)] = previous_bias
