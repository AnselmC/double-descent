import argparse
import time
import os
import json
import torch
import torchvision.datasets as datasets
import numpy as np
import cupy as cp
from helpers import get_transform


def dot(a, b, cuda=False):
    if cuda:
        res = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.dot(a, b)

def maximum(a, cuda=False):
    if cuda:
        res = cp.maximum(a, 0)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.maximum(a, 0)

def norm(a, axis, cuda=False):
    if cuda:
        res = cp.linalg.norm(a, axis=axis)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.linalg.norm(a, axis=axis)

def randn(a, b, cuda=False):
    if cuda:
        res = cp.random.randn(a, b)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.random.randn(a, b)

def cos(a, cuda=False):
    if cuda:
        res = cp.cos(a)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.cos(a)

def lstsq(x, y, cuda=False):
    if cuda:
        x_inv = cp.linalg.pinv(x)
        res = cp.dot(x_inv, y)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.linalg.lstsq(x, y, rcond=None)[0]

def zero_one(y, target, cuda=False):
    if cuda:
        res = (target != cp.around(y)).sum() / len(target)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return (target != np.around(y)).sum() / len(target)

def compute_random_fourier_features(num_params, input_train, target_train, input_test, target_test, relu=False, cuda=False):
    v = generate_random_fourier_matrix(num_params, relu, cuda)
    x = transform_inputs(v, input_train, relu, cuda)
    a = lstsq(x, target_train, cuda)
    preds = dot(x, a, cuda)
    mse_train = mse(preds, target_train, cuda)
    zero_one_train = zero_one(preds, target_train, cuda)
    x = transform_inputs(v, input_test, relu)
    preds = dot(x, a, cuda)
    mse_test = mse(preds, target_test)
    zero_one_test = zero_one(preds, target_test, cuda)
    a_norm = norm(a, axis=0, cuda=cuda)

    return (mse_train, zero_one_train), (mse_test, zero_one_test), a_norm, a, v


def generate_random_fourier_matrix(num_params, relu, cuda):
    if relu:
        v = randn(num_params, 784, cuda)
        v /= norm(v, axis=0, cuda=cuda)
    else:
        v = randn(num_params, 784, cuda) + 1/25
    return v


def transform_inputs(random_fourier_matrix, inputs, relu=False, cuda=False):
    transformed = dot(inputs, random_fourier_matrix.T, cuda)
    if relu:
        transformed = maximum(transformed, cuda)
    else:
        transformed = cos(transformed, cuda)
    return transformed


def mse(predictions, targets, cuda=False):
    if cuda:
        res = cp.square(cp.subtract(predictions, targets)).mean()
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.square(np.subtract(predictions, targets)).mean()


def get_data(train=True, cuda=False):
    subset_size = int(1e4)

    dataset = datasets.MNIST(root='./data',
                             download=True,
                             train=train,
                             transform=get_transform("mnist"))

    inputs = []
    targets = []

    for i in range(subset_size):
        #if cuda:
        #    inputs.append(cp.asarray(dataset[i][0]))
        #    targets.append(cp.asarray(dataset[i][1]))

        inputs.append(np.array(dataset[i][0]))
        targets.append(np.array(dataset[i][1]))

    inputs = np.array(inputs)
    targets = np.array(targets)
    if cuda:
        inputs = cp.asarray(inputs)
        targets = cp.asarray(targets)

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

    parser.add_argument("--cuda", action="store_true", help="Whether to use cuda or not (default: False)", default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    num_params = config["num_params"]["random_fourier"]
    relu = args.relu
    cuda = args.cuda
    mse_train_losses = []
    zero_one_train_losses = []
    mse_test_losses = []
    zero_one_test_losses = []
    norms = []

    input_train, target_train = get_data(cuda=cuda)
    input_test, target_test = get_data(train=False, cuda=cuda)

    for num in num_params:
        start = time.time()
        train_loss, test_loss, a_norm, features, transforms = compute_random_fourier_features(
            num, input_train, target_train, input_test, target_test, relu=relu, cuda=cuda)
        print("Model with {} params took {:.4f}s".format(num, time.time()-start))
        mse_train_losses.append(train_loss[0])
        zero_one_train_losses.append(train_loss[1])
        mse_test_losses.append(test_loss[0])
        zero_one_test_losses.append(test_loss[1])
        norms.append(a_norm)
        model_path = os.path.join("data", "models", "rff")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = str(num)
        if cuda:
            model_name += "_cuda"
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
    if cuda:
        name += "_cuda"
    file_name = os.path.join(results_path, name + ".json")
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)

