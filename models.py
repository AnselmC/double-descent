import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.optim.lr_scheduler import MultiStepLR


def get_model_by_name(model_name):
    models = [TwoLayerNN, RandomFourier]
    for model in models:
        if model.name() == model_name:
            return model


def cross_entropy():
    return nn.CrossEntropyLoss()


def zero_one():
    return ZeroOneLoss()


def mse_loss():
    return MSELoss()


class ZeroOneLoss:
    def __call__(model, output, labels):
        raise NotImplementedError("Zero-one loss has not yet been implemented")


class MSELoss:
    def __call__(model, outputs, labels):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        outputs_soft = F.softmax(outputs, dim=0).to(device)
        one_hot_labels = torch.FloatTensor(len(labels), outputs.shape[1]).to(device)
        one_hot_labels.zero_()
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        loss = torch.mean((outputs_soft - one_hot_labels) ** 2)
        return loss


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self._train_step = 0
        self._val_step = 0
        self._test_step = 0

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def epoch_step(self):
        self._train_step = 0
        self._val_step = 0

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def train_step(self, data, device):
        raise NotImplementedError()

    def val_step(self, data, device):
        raise NotImplementedError()

    def test_step(self, data, device):
        raise NotImplementedError()

    def init_weights(self, prev_model):
        pass


class RandomFourier(CustomModel):
    def __init__(self, dim_input, num_hidden, dim_output):
        super(RandomFourier, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden, bias=False)
        nn.init.normal_(self.hidden.weight, std=0.04)
        self.hidden.weight.requires_grad = False  # fix weights
        self.out = nn.Linear(num_hidden, dim_output, bias=False)
        self._optimizer = optimizers.SGD(
            self.parameters(), lr=0.001, momentum=0.95)
        self._loss_function = nn.MSELoss()
        self._one_hot_label = None
        self._dim_output = dim_output

    def forward(self, x):
        x = torch.cos(self.hidden(x))  # exp(i * x) = cos(x) (+ i*sin(x))
        x = nn.Softmax(self.out(x))
        return x

    def train_step(self, data, device):
        self._train_step += 1
        if self._train_step == 1:
            self.train()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        one_hot_labels = torch.FloatTensor(len(labels), self._dim_output)
        one_hot_labels.zero_()
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        self._optimizer.zero_grad()
        loss = self._loss_function(outputs.dim, one_hot_labels)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def val_step(self, data, device):
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        one_hot_labels = torch.FloatTensor(len(labels), self._dim_output)
        one_hot_labels.zero_()
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        loss = self._loss_function(outputs.dim, one_hot_labels)
        return loss.item()

    def test_step(self, data, device):
        self._test_step += 1
        if self._test_step == 1:
            self.eval()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        one_hot_labels = torch.FloatTensor(len(labels), self._dim_output)
        one_hot_labels.zero_()
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
        loss = self._loss_function(outputs.dim, one_hot_labels)
        test_loss = loss.item()
        _, predicted = torch.max(outputs.dim, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct/total
        return test_loss, accuracy

    @staticmethod
    def name():
        return "random_fourier"


class TwoLayerNN(CustomModel):
    def __init__(self, dim_input, num_hidden, dim_output, loss_fn="cross_entropy"):
        super(TwoLayerNN, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self.out = nn.Linear(num_hidden, dim_output)
        self._dim_output = dim_output
        self._subset_size = 4000
        self._loss_function = loss_fn
        self._optimizer = optimizers.SGD(
            self.parameters(), lr=0.001, momentum=0.95)
        self.scheduler = MultiStepLR(self._optimizer, milestones=list(
            range(500, 6000, 500)), gamma=0.9)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

    def init_weights(self, prev_model):
        if prev_model is None or self.num_parameters() > self._subset_size * self._dim_output:
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
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        loss = self._loss_function(outputs, labels)
        return loss.item()

    def test_step(self, data, device):
        self._test_step += 1
        if self._test_step == 1:
            self.eval()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        loss = self._loss_function(outputs, labels)
        test_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct/total
        return test_loss, accuracy

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
