import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from collections import namedtuple
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


class ProgressBar:
    def __init__(self, max_epochs):
        self._max_epochs = max_epochs
        self._iteration = 0
        self._progress_string = "\rNet: {}\t|Epoch: {}\t|{} %\t|Train/val loss: {:.2f}/{:.2f}|\t"

    def done(self, net, train_loss, val_loss, test_loss, acc):
        _, width = os.popen("stty size", "r").read().split()

        finished_string = "=" * int(width)
        finished_string += "\nFinished training model with {} parameters".format(
            net.num_parameters())
        finished_string += "\nFinal train/val/test loss: {:.2f}/{:.2f}/{:.2f}".format(
            train_loss, val_loss, test_loss)
        finished_string += "\nAccuracy: {}\n".format(acc)
        print(finished_string, end="\r")

    def update(self, net, train_loss, val_loss):
        self._iteration += 1
        percentage_done = int(100 * self._iteration/self._max_epochs)
        current_progress_string = self._progress_string.format(net,
                                                               self._iteration,
                                                               percentage_done,
                                                               train_loss,
                                                               val_loss)

        _, width = os.popen("stty size", "r").read().split()
        # 2 for opening and closing brackets
        #width = 70
        available_width = max(
            0, int(width) - len(current_progress_string.expandtabs()) - 2)

        progress_width = int(available_width * percentage_done/100)
        progress_bar = "[" + "#" * progress_width + \
            " " * (available_width - progress_width) + "]"
        current_progress_string += progress_bar
        print(current_progress_string, end="\r")
