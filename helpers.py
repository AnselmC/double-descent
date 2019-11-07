import os
import time
import datetime
from collections import namedtuple

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler

Dataset = namedtuple("Dataset", ["train_loader", "val_loader", "test_loader"])


def get_dataset(dataset, train_subset_size, batch_size):
    if dataset.lower() == "mnist":
        ds = datasets.MNIST
    else:
        raise NotImplementedError("Dataset {} isn't available".format(dataset))

    transform = get_transform(dataset)

    train_set = ds(root='./data',
                   download=True,
                   train=True,
                   transform=transform)
    val_subset_size = int(0.2 * train_subset_size)
    random_train_indices = np.random.randint(len(train_set),
                                             size=train_subset_size)
    random_val_indices = np.random.choice([x for x in range(len(train_set))
                                           if x not in random_train_indices],
                                          size=val_subset_size)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=4,
                              sampler=sampler.SubsetRandomSampler(random_train_indices))
    val_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            num_workers=4,
                            sampler=sampler.SubsetRandomSampler(random_val_indices))
    test_set = datasets.MNIST(root='./data',
                                   download=True,
                                   train=False,
                                   transform=transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2)

    dataset = Dataset(train_loader=train_loader,
                      val_loader=val_loader, test_loader=test_loader)
    return dataset


def get_transform(dataset):
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda img: img.reshape(-1))])

    return transform


def init_print(num_models, model_name, dataset_name):
    _, width = os.popen("stty size", "r").read().split()
    print("=" * int(width))
    print("Training {} {} models on {} dataset...".format(
        num_models, model_name, dataset_name))

    print("=" * int(width))


class ProgressBar:
    def __init__(self, num_models, num_epochs, num_batches):
        self._num_models = num_models
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._model = 0
        self._epoch = 0
        self._batch = 0
        self._val_loss = 0
        self._train_loss = 0
        self._start_time_global = time.time()
        self._start_time_model = time.time()
        self._time_per_epoch = 0
        self._eta = "Estimating..."
        self._progress_string = "\rNet: {} | Epoch: {:4d} | Batch: {:3d} | {:.2f} % | {} | Train/val: {:.2f}/{:.2f} |"

    def finished_model(self, num_parameters, test_loss, acc):
        _, width = os.popen("stty size", "r").read().split()
        finished_string = "=" * int(width)
        finished_string += "\nFinished training model with {} parameters".format(
            num_parameters)
        finished_string += "\nFinal train/val/test loss: {:.2f}/{:.2f}/{:.2f}".format(
            self._train_loss, self._val_loss, test_loss)
        finished_string += "\nAccuracy: {}".format(acc)
        training_time = datetime.timedelta(
            seconds=round(time.time()-self._start_time_model))
        finished_string += "\nTraining model took: {}\n".format(training_time)
        print(finished_string)

    def finished_training(self):
        _, width = os.popen("stty size", "r").read().split()
        finished_string = "=" * int(width)
        finished_string += "\nFinished entire training of {} models.".format(
            self._num_models)
        training_time = datetime.timedelta(
            seconds=round(time.time()-self._start_time_global))
        finished_string += "\n Training took {}\n".format(training_time)
        finished_string += "=" * int(width)
        print(finished_string, end="\r")

    def update_batch(self):
        self._batch += 1
        self._print_progress_string()

    def update_epoch(self, train_loss, val_loss):
        self._batch = 0
        self._epoch += 1
        if self._epoch == 1:
            self._time_per_epoch = time.time() - self._start_time_model
        secs_left = self._time_per_epoch * (self._num_epochs - self._epoch)
        self._eta = datetime.timedelta(seconds=round(secs_left))
        self._val_loss = val_loss
        self._train_loss = train_loss
        self._print_progress_string()

    def update_model(self):
        self._model += 1
        self._epoch = 0
        if self._model == 0:
            self._start_time_global = time.time()
        self._start_time_model = time.time()
        self._print_progress_string()

    def _print_progress_string(self):
        # percent epochs = current_epoch / num_epochs
        # percent batches = current batch / num_batches
        # percent total = current_batch_total / num_batches_total
        # num_batches_total = num_epochs * num_batches
        # current_batch_total = (current_epoch-1) * num_batches + current_batches
        percentage_done = 100 * (self._epoch * self._num_batches +
                                 self._batch)/(self._num_epochs*self._num_batches)
        current_progress_string = self._progress_string.format(self._model,
                                                               self._epoch,
                                                               self._batch,
                                                               percentage_done,
                                                               self._eta,
                                                               self._train_loss,
                                                               self._val_loss)
        _, width = os.popen("stty size", "r").read().split()
        available_width = max(
            0, int(width) - len(current_progress_string.expandtabs()) - 2)

        progress_width = int(available_width * percentage_done/100)
        progress_bar = "[" + "#" * progress_width + \
            " " * (available_width - progress_width) + "]"
        current_progress_string += progress_bar
        print(current_progress_string, end="\r")
