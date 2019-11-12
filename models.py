import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.optim.lr_scheduler import MultiStepLR


def get_model_by_name(model_name):
    models = [TwoLayerNN, RandomFourierCluster, RandomFourierReLUCluster]
    for model in models:
        if model.name() == model_name:
            return model


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


class RandomFourierReLUCluster:
    pass


class RandomFourierReLUSingle(CustomModel):
    pass


class RandomFourierCluster:
    def __init__(self, dim_input, num_hidden, dim_output):
        self._classifiers = []
        self._train_step = 0
        self._val_step = 0
        self._test_step = 0
        self._num_parameters = num_hidden
        # one-vs-all
        for i in range(dim_output):
            self._classifiers.append(
                RandomFourierSingle(dim_input, num_hidden, i))

    def save(self, file_name):
        for clf in self._classifiers:
            clf.save(file_name + "_" + str(clf.cls))

    def num_parameters(self):
        return self._num_parameters

    def init_weights(self, prev_model):
        pass

    def to(self, device):
        for clf in self._classifiers:
            clf.to(device)

    def epoch_step(self):
        self._train_step = 0
        self._val_step = 0

    def train_step(self, data, device):
        # train each classifier, return average loss
        loss = 0.0
        for clf in self._classifiers:
            loss += clf.train_step(data, device)

        return loss / len(self._classifiers)

    def val_step(self, data, device):
        # train each classifier, return average loss
        loss = 0.0
        for clf in self._classifiers:
            loss += clf.train_step(data, device)

        return loss / len(self._classifiers)

    def test_step(self, data, device):
        # train each classifier, return average loss
        running_loss = 0.0
        running_acc = 0.0
        for clf in self._classifiers:
            loss, acc = clf.test_step(data, device)
            running_loss += loss
            running_acc += acc
            print("Accuracy: {}".format(acc))

        return running_loss / len(self._classifiers), running_acc / len(self._classifiers)

    @staticmethod
    def name():
        return "random_fourier"


class RandomFourierSingle(CustomModel):
    def __init__(self, dim_input, num_hidden, cls):
        super(RandomFourierSingle, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self._init_from_n_dim_unit_sphere(num_hidden, dim_input)
        self.hidden.weight.requires_grad = False  # fix weights
        self.out = nn.Linear(num_hidden, 1)
        self._optimizer = optimizers.SGD(
            self.parameters(), lr=0.001, momentum=0.95)
        self.cls = cls
        self._loss_function = nn.MSELoss()

    def _init_from_n_dim_unit_sphere(self, num_points, dim):
        with torch.no_grad():
            weights = torch.randn(dim, num_points)
            weights /= torch.norm(weights, dim=0)
            self.hidden.weight = nn.Parameter(weights.T)

    def forward(self, x):
        x = torch.cos(self.hidden(x))  # exp(i * x) = cos(x) (+ i*sin(x))
        x = torch.sigmoid(self.out(x))
        return x

    def train_step(self, data, device):
        self._train_step += 1
        if self._train_step == 1:
            self.train()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        labels = (labels == self.cls).float()
        self._optimizer.zero_grad()
        outputs = self(inputs)
        loss = self._loss_function(outputs.view(-1), labels) + \
            self.out.weight.norm(2)  # + l2_reg for a_i
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def val_step(self, data, device):
        inputs, labels = data[0].to(
            device), data[1].to(device)
        labels = (labels == self.cls).float()
        outputs = self(inputs)
        loss = self._loss_function(outputs.view(-1), labels) + self.out.weight.norm(2)
        return loss.item()

    def test_step(self, data, device):
        self._test_step += 1
        if self._test_step == 1:
            self.eval()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        labels = (labels == self.cls).float()
        outputs = self(inputs)
        loss = self._loss_function(outputs.view(-1), labels)
        test_loss = loss.item()
        predicted = torch.round(outputs)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct/total

        return test_loss, accuracy, outputs


class TwoLayerNN(CustomModel):
    def __init__(self, dim_input, num_hidden, dim_output, loss_fn="cross_entropy"):
        super(TwoLayerNN, self).__init__()
        self.hidden = nn.Linear(dim_input, num_hidden)
        self.out = nn.Linear(num_hidden, dim_output)
        self._loss_function = TwoLayerNN.mse_loss
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer = optimizers.SGD(
            self.parameters(), lr=0.001, momentum=0.95)
        self.scheduler = MultiStepLR(self._optimizer, milestones=list(
            range(500, 6000, 500)), gamma=0.9)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

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

    @staticmethod
    def get_loss_fn_by_name(name):
        losses = {"cross_entropy": nn.CrossEntropyLoss,
                  "squared": TwoLayerNN.mse_loss,
                  "zero_one": TwoLayerNN.zero_one_loss}
        return losses.get(name)

    @staticmethod
    def zero_one_loss(output, labels):
        pass

    @staticmethod
    def mse_loss(outputs, labels):
        print(outputs)
        indices = [(col, int(row)) for col, row in enumerate(labels)]
        outputs = torch.tensor(
            [outputs[elem] for elem in indices], dtype=torch.float, requires_grad=True)
        ones = torch.ones(outputs.shape)
        # print(outputs)
        loss = torch.mean((outputs - ones) ** 2)
        return loss

    def test_step(self, data, device):
        self._test_step += 1
        if self._test_step == 1:
            self.eval()
        inputs, labels = data[0].to(
            device), data[1].to(device)
        outputs = self(inputs)
        loss = self._loss_function(outputs, labels)
        test_loss = loss.item()
        predicted, _ = torch.max(outputs.data, 1)
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
