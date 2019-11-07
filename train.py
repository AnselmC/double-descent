# built-in
import json
import time
import os
import argparse

# 3rd party
import torch
import torchvision
import torch.nn as nn
import torch.optim as optimizers
from torch.optim.lr_scheduler import MultiStepLR

# own
from nets import initialize_with_previous_weights, initialize_with_previous_bias, get_model_by_name
from helpers import get_dataset, ProgressBar

AVAILABLE_NETS = ["two_layer_nn"]


class Training:
    def __init__(self, model, dataset, num_params, epochs, prev_net=None):
        self._train_loader = dataset.train_loader
        self._val_loader = dataset.val_loader
        self._test_loader = dataset.test_loader
        num_classes = len(self._train_loader.dataset.classes)
        self._interpolation_threshold = num_classes * \
            len(self._train_loader.dataset)
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self._all_models = []
        self._epochs = epochs
        self.all_train_losses = {}
        self.all_val_losses = {}
        self.all_test_losses = {}
        self._loss_function = nn.CrossEntropyLoss()
        self._dataset_name = type(self._val_loader.dataset).__name__
        self._model_name = model.name()
        input_dim = self._val_loader.dataset[0][0].size()[0]
        if prev_net is not None:
            prev_num_hidden = len(prev_net['hidden.weight'])
            self._prev_net = model(input_dim, prev_num_hidden, num_classes)
            self._prev_net.load_state_dict(prev_net)
            num_params = [num_params[num_params.index(prev_num_hidden) + 1]]
        else:
            self._prev_net = None

        for p in num_params:
            self._all_models.append(model(input_dim, p, num_classes))

    def save(self):
        path = os.path.join("data", "results", "epochs_" + str(self._epochs))
        file_name = os.path.join(path, self._model_name + ".json")
        if not os.path.exists(path):
            os.makedirs(path)
        content = {}
        content["Train losses"] = self.all_train_losses
        content["Val losses"] = self.all_val_losses
        content["Test losses"] = self.all_test_losses
        with open(file_name, "w") as fd:
            json.dump(content, fd)

    def start(self):
        try:
           _, _ = os.popen("stty size", "r").read().split()
           run_from_term = True
        except Exception:
            run_from_term = False
        pbar = ProgressBar(len(self._all_models), self._epochs, len(self._train_loader), run_from_term)
        pbar.init_print(len(self._all_models), self._model_name, self._dataset_name)
        for i, net in enumerate(self._all_models):
            pbar.update_model()
            if self._prev_net is None:
                # initialize smallest net using Xavier Glorot-uniform distribution
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_uniform_(net.hidden.weight, gain=gain)
                # or only for hidden?
                nn.init.xavier_uniform_(net.out.weight, gain=gain)
            else:
                nn.init.normal_(net.hidden.weight, 0, 0.01)
                initialize_with_previous_weights(
                    net.hidden.weight, self._prev_net.hidden.weight)
                initialize_with_previous_bias(
                    net.hidden.bias, self._prev_net.hidden.bias)
                nn.init.normal_(net.out.weight, 0, 0.01)

            net.to(self._device)
            # init learning rate not clear
            optimizer = optimizers.SGD(
                net.parameters(), lr=0.001, momentum=0.95)
            scheduler = MultiStepLR(optimizer, milestones=list(
                range(500, 6000, 500)), gamma=0.9)
            train_losses = []
            val_losses = []

            for epoch in range(self._epochs):
                running_loss = 0.0
                for data in self._train_loader:
                    pbar.update_batch()
                    inputs, labels = data[0].to(
                        self._device), data[1].to(self._device)
                    #inputs, labels = data
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = self._loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_losses.append(running_loss / len(self._train_loader))

                if net.num_parameters() < self._interpolation_threshold:
                    scheduler.step()

                net.eval()
                running_val_loss = 0.0
                for data in self._val_loader:
                    images, labels = data[0].to(
                        self._device), data[1].to(self._device)
                    outputs = net(images)
                    loss = self._loss_function(outputs, labels)
                    running_val_loss += loss.item()
                val_losses.append(running_val_loss / len(self._val_loader))

                pbar.update_epoch(train_losses[-1], val_losses[-1])

                if train_losses[-1] == 0.0 and net.num_parameters() < self._interpolation_threshold:
                    break

            net_name = self._model_name + "_" + str(net.num_parameters())
            path = os.path.join("data", "nets", "epochs_" + str(self._epochs))
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = os.path.join(path, net_name)
            self.all_train_losses[net_name] = train_losses
            self.all_val_losses[net_name] = val_losses
            net.eval()
            running_test_loss = 0.0
            total = 0
            correct = 0
            for data in self._test_loader:
                images, labels = data[0].to(
                    self._device), data[1].to(self._device)
                outputs = net(images)
                loss = self._loss_function(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_loss = running_test_loss / len(self._test_loader)
            pbar.finished_model(net.num_parameters(), test_loss, correct/total)
            print("Saving network to \"{}\"\n".format(file_name))
            self._prev_net = net
            self.save()
            torch.save(net.state_dict(), file_name)
            self.all_test_losses[net_name] = test_loss

        pbar.finished_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple trainings "
                                     "with growing network capacity")

    parser.add_argument(
        "--model", type=int, help="select model to train: 0 [two_layer_net], 1 [Random Fourier Features], 2 [ReLU Random Fourier Features]", choices={0, 1, 2}, dest="model", default=0)

    parser.add_argument("--prev", type=str, help="run single model size given path to model of previous size", dest="prev")
    
    parser.add_argument(
        "--config", type=str, help="config file for training", required=True, dest="config")
    args = parser.parse_args()
    with open(args.config, "r") as fd:
        config = json.load(fd)
    dataset_name = "mnist"
    model_name = AVAILABLE_NETS[args.model]
    batch_size = config["batch_size"][dataset_name.lower()]
    train_subset_size = config["train_subset_size"][dataset_name.lower()]
    epochs = config["epochs"]
    previous_net = None
    if args.prev:
        if not os.path.exists(args.prev):
            raise FileNotFoundError("Couldn't find {}".format(args.prev))
        previous_net = torch.load(args.prev)
    num_params = config["num_params"][model_name]
    model = get_model_by_name(model_name)
    dataset = get_dataset(dataset_name, train_subset_size, batch_size)

    training = Training(model, dataset, num_params, epochs, prev_net=previous_net)
    training.start()
    training.save()
