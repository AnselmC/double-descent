import torch
import torchvision
import torch.nn as nn
import argparse
import json
import time
import os

import torch.optim as optimizers
from torch.optim.lr_scheduler import MultiStepLR
from nets import initialize_with_previous_weights, initialize_with_previous_bias
import helpers
from helpers import ProgressBar
from helpers import get_dataset, ProgressBar
from nets import get_model_by_name

AVAILABLE_NETS = ["two_layer_nn"]


class Training:
    def __init__(self, model, dataset, num_params, epochs):
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
        for p in num_params:
            self._all_models.append(model(input_dim, p, num_classes))

    def save(self):
        file_name = os.path.join("data", "results", self._model_name + ".json")
        content = {}
        content["Train losses"] = self.all_train_losses
        content["Val losses"] = self.all_val_losses
        content["Test losses"] = self.all_test_losses
        with open(file_name, "w") as fd:
            json.dump(content, fd)

        
    def start(self):
        print("=" * 30)
        print("TRAINING {} NETS ON {} DATASET USING {} MODEL".format(
            len(self._all_models), self._dataset_name.upper(), self._model_name.upper(), ))
        for i, net in enumerate(self._all_models):
            start = time.time()
            pbar = ProgressBar(self._epochs)
            if i == 0:
                # initialize smallest net using Xavier Glorot-uniform distribution
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_uniform_(net.hidden.weight, gain=gain)
                # or only for hidden?
                nn.init.xavier_uniform_(net.out.weight, gain=gain)
            else:
                nn.init.normal_(net.hidden.weight, 0, 0.01)
                initialize_with_previous_weights(
                    net.hidden.weight, self._all_models[i-1].hidden.weight)
                initialize_with_previous_bias(
                    net.hidden.bias, self._all_models[i-1].hidden.bias)
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
                    inputs, labels = data[0].to(self._device), data[1].to(self._device)
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
                    images, labels = data[0].to(self._device), data[1].to(self._device)
                    outputs = net(images)
                    loss = self._loss_function(outputs, labels)
                    running_val_loss += loss.item()
                val_losses.append(running_val_loss / len(self._val_loader))

                pbar.update(i+1, train_losses[-1], val_losses[-1])

                if train_losses[-1] == 0.0 and net.num_parameters() < self._interpolation_threshold:
                    break

            net_name = self._model_name + "_" + str(net.num_parameters())
            path = os.path.join("data", "nets", net_name)
            self.all_train_losses[net_name] = train_losses
            self.all_val_losses[net_name] = val_losses
            net.eval()
            running_test_loss = 0.0
            total = 0
            correct = 0
            for data in self._test_loader:
                images, labels = data[0].to(self._device), data[1].to(self._device)
                outputs = net(images)
                loss = self._loss_function(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_loss = running_test_loss / len(self._test_loader)
            pbar.done(net, train_losses[-1], val_losses[-1], test_loss, correct/total)
            print("Saving network to \"{}\"\n".format(path), end="\r")
            print("Training took {:.2f}s".format(time.time()-start))
            torch.save(net.state_dict(), path)
            self.all_test_losses[net_name] = test_loss


# Run different sized RFF/ReLU RFF/TwoLayerNN on MNIST
# varying model, varying size
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple trainings "
                                     "with growing network capacity")

    parser.add_argument(
        "--model", type=int, help="select model to train: 0 [two_layer_net], 1 [Random Fourier Features], 2 [ReLU Random Fourier Features]", choices={0, 1, 2}, dest="model", default=0)

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
    num_params = config["num_params"][model_name]
    model = get_model_by_name(model_name)
    dataset = get_dataset(dataset_name, train_subset_size, batch_size)

    training = Training(model, dataset, num_params, epochs)
    training.start()
    training.save()
