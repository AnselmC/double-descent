import argparse
import os
import json
import torch
import torchvision.datasets as datasets
import numpy as np
from helpers import get_transform


def compute_random_fourier_features(num_params, input_train, target_train, input_test, target_test, relu=False):
    v = generate_random_fourier_matrix(num_params, relu)
    x = transform_inputs(v, input_train, relu)
    a, _, _, _ = np.linalg.lstsq(x, target_train, rcond=None)
    preds = x@a
    mse_train = mse(preds, target_train)
    zero_one_train = (target_train != np.round(preds)).sum()/len(target_train)
    x = transform_inputs(v, input_test, relu)
    preds = x@a
    #import pdb;pdb.set_trace()
    mse_test = mse(preds, target_test)
    zero_one_test = (target_test != np.round(preds)).sum()/len(target_test)
    a_norm = np.linalg.norm(a)

    return (mse_train, zero_one_train), (mse_test, zero_one_test), a_norm, a, v


def generate_random_fourier_matrix(num_params, relu):
    if relu:
        v = np.random.randn(num_params, 784)
        v /= np.linalg.norm(v, axis=0)
    else:
        v = np.random.randn(num_params, 784) + 1/25
    return v


def transform_inputs(random_fourier_matrix, inputs, relu=False):
    transformed = inputs @ random_fourier_matrix.T
    if relu:
        transformed = np.maximum(transformed, 0)
    else:
        transformed = np.cos(transformed)
    return transformed


def mse(predictions, targets):
    return np.square(np.subtract(predictions, targets)).mean()


def get_data(train=True):
    subset_size = int(1e4)

    dataset = datasets.MNIST(root='./data',
                             download=True,
                             train=train,
                             transform=get_transform("mnist"))

    inputs = []
    targets = []

    for i in range(subset_size):
        inputs.append(np.array(dataset[i][0]))
        targets.append(np.array(dataset[i][1]))

    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets


# apply RFF transformation with variable amount of parameters
# compute params using pseudo-inverse
# compute squared-loss on test set
# QUESTION: how would you compute zero-one loss in this particular case??
# compute squared-loss on train set
# compute l2 norm of params

# do the same for ReLU RFF
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RFF or ReLu RFF using linear regression")
    parser.add_argument(
        "--config", type=str, dest="config", help="json file that holds config information", required=True)

    parser.add_argument("--relu", action="store_true", help="Whether to use relu or not (default: False)", default=False)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    num_params = config["num_params"]["random_fourier"]
    relu = args.relu
    mse_train_losses = []
    zero_one_train_losses = []
    mse_test_losses = []
    zero_one_test_losses = []
    norms = []

    input_train, target_train = get_data()
    input_test, target_test = get_data(train=False)

    for num in num_params:
        print(num, end="\r")
        train_loss, test_loss, norm, features, transforms = compute_random_fourier_features(
            num, input_train, target_train, input_test, target_test, relu=relu)
        mse_train_losses.append(train_loss[0])
        zero_one_train_losses.append(train_loss[1])
        mse_test_losses.append(test_loss[0])
        zero_one_test_losses.append(test_loss[1])
        norms.append(norm)
        model_path = os.path.join("data", "models", "rff")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = str(num)
        features_fname = os.path.join(model_path, model_name + "_features")
        np.savez(features_fname, features)
        transforms_fname = os.path.join(model_path, model_name + "_transforms")
        np.savez(transforms_fname, transforms)


    results = {}
    results["norms"] = norms
    results["mse_train_losses"] = mse_train_losses
    results["zero_one_train_losses"] = zero_one_train_losses
    results["mse_test_losses"] = mse_test_losses
    results["zero_one_test_losses"] = zero_one_test_losses

    results_path = os.path.join("data", "results", "rff")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    name = "relu" if relu else "reg"
    file_name = os.path.join(results_path, name + ".json")
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)

