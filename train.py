# built-in
import json
import time
import os
import argparse

# 3rd party
import torch

# own
from models import get_model_by_name, cross_entropy, mse_loss, zero_one
from helpers import get_dataset, Progress

AVAILABLE_MODELS = ["two_layer_nn", "random_fourier"]

AVAILABLE_LOSS_FNS = [
    mse_loss(),
    cross_entropy(),
    zero_one()
]


class Training:
    def __init__(self, model, dataset, num_params, epochs, loss_fn, prev_model=None, num_models=None):
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
        self._loss_name = type(loss_fn).__name__
        self.all_train_losses = {}
        self.all_val_losses = {}
        self.all_test_losses = {}
        self._dataset_name = type(self._val_loader.dataset).__name__
        self._model_name = model.name()
        input_dim = self._val_loader.dataset[0][0].size()[0]
        if prev_model is not None:
            prev_num_hidden = len(prev_model['hidden.weight'])
            self._prev_model = model(
                input_dim, prev_num_hidden, num_classes, loss_fn)
            self._prev_model.load_state_dict(prev_model)
            num_params = num_params[num_params.index(prev_num_hidden) + 1:]
        else:
            self._prev_model = None

        if num_models is not None:
            num_params = num_params[:num_models]

        for p in num_params:
            self._all_models.append(model(input_dim, p, num_classes, loss_fn))

    def save(self):
        path = os.path.join("data", "results",
                            self._model_name, self._loss_name)

        file_name = os.path.join(path, str(self._epochs) + ".json")
        if not os.path.exists(path):
            os.makedirs(path)
        content = {}
        content["Train losses"] = self.all_train_losses
        content["Val losses"] = self.all_val_losses
        content["Test losses"] = self.all_test_losses
        if os.path.exists(file_name):
            with open(file_name, "r") as fd:
                old_content = json.load(fd)
            for loss_key in old_content.keys():
                for model_key in old_content[loss_key].keys():
                    content[loss_key][model_key] = old_content[loss_key][model_key]
        with open(file_name, "w") as fd:
            json.dump(content, fd, indent=4, sort_keys=True)

    def start(self):
        try:
            _, _ = os.popen("stty size", "r").read().split()
            run_from_term = True
        except Exception:
            run_from_term = False
        progress = Progress(len(self._all_models), self._epochs, len(
            self._train_loader), run_from_term)
        progress.init_print(len(self._all_models),
                            self._model_name, self._dataset_name)

        for model in self._all_models:  # different sized models
            progress.update_model()
            model.init_weights(self._prev_model)  # need for 2layer NN
            model.to(self._device)
            train_losses = []
            val_losses = []
            for epoch in range(epochs):
                model.epoch_step()
                running_loss = 0.0
                for data in self._train_loader:
                    progress.update_batch()
                    running_loss += model.train_step(data, self._device)
                train_losses.append(running_loss / len(self._train_loader))

                running_loss = 0.0
                for data in self._val_loader:
                    running_loss += model.val_step(data, self._device)
                val_losses.append(running_loss / len(self._val_loader))

                progress.update_epoch(train_losses[-1], val_losses[-1])

                # TODO: only for two layer nn?
                if train_losses[-1] == 0.0 and model.num_parameters() < self._interpolation_threshold:
                    break

            running_loss = 0
            running_acc = 0
            for data in self._test_loader:
                loss, acc = model.test_step(data, self._device)
                running_loss += loss
                running_acc += acc
            accuracy = running_acc / len(self._test_loader)
            test_loss = running_loss / len(self._test_loader)

            progress.finished_model(
                model.num_parameters(), test_loss, accuracy)

            model_name = str(model.num_parameters())
            self.all_test_losses[model_name] = test_loss
            self.all_train_losses[model_name] = train_losses
            self.all_val_losses[model_name] = val_losses
            self._prev_model = model

            path = os.path.join(
                "data", "models", self._model_name, self._loss_name, "epochs_" + str(self._epochs))
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = os.path.join(path, model_name)
            self.save()
            print("Saving model to \"{}\"\n".format(file_name))
            model.save(file_name)
        progress.finished_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple trainings "
                                     "with growing network capacity")

    parser.add_argument(
        "--model", type=int, help="select model to train: 0 [two_layer_net], 1 [Random Fourier Features], 2 [ReLU Random Fourier Features]", choices={0, 1, 2}, dest="model", default=0)

    parser.add_argument(
        "--prev", type=str, help="run single model size given path to model of previous size", dest="prev")

    parser.add_argument(
        "--num", type=int, help="the number of model sizes to run, starting at smallest model or model given with --prev flag", dest="num")

    parser.add_argument(
        "--config", type=str, help="config file for training", required=True, dest="config")

    parser.add_argument(
        "--loss", type=int, help="The loss function to use 0 [Squared](default), 1 [Zero-one], 2 [Cross-Entropy]", dest="loss", default=0, choices={0, 1, 2})

    parser.add_argument(
        "--epochs", type=int, help="The number of epochs to train, if not given, the value in the config file is taken", dest="epochs")

    args = parser.parse_args()

    with open(args.config, "r") as fd:
        config = json.load(fd)

    dataset_name = "mnist"
    model_name = AVAILABLE_MODELS[args.model]
    loss_fn = AVAILABLE_LOSS_FNS[args.loss]
    batch_size = config["batch_size"][dataset_name.lower()]
    train_subset_size = config["train_subset_size"][model_name.lower()]
    epochs = args.epochs if args.epochs else config["epochs"]
    previous_model = None
    if args.prev:
        if loss_fn not in args.prev:
            raise RuntimeError("Use same loss function")
        if not os.path.exists(args.prev):
            raise FileNotFoundError("Couldn't find {}".format(args.prev))
        previous_model = torch.load(args.prev)
    num_params = config["num_params"][model_name]
    num_models = None
    if args.num:
        num_models = args.num
    model = get_model_by_name(model_name)
    dataset = get_dataset(dataset_name, train_subset_size, batch_size)

    training = Training(model, dataset, num_params,
                        epochs, prev_model=previous_model, num_models=num_models, loss_fn=loss_fn)
    training.start()
    training.save()
