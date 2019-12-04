from __future__ import print_function

import argparse
import math
import time
import os
import json
import torch
import torchvision.datasets as datasets
import numpy as np
import sklearn
from sklearn.metrics import zero_one_loss

try:
    import cupy as cp
except:
    pass
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


def norm(a, cuda=False):
    if cuda:
        res = cp.linalg.norm(a)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.linalg.norm(a, axis=0).mean()


def randn(a, b, cuda=False):
    if cuda:
        res = cp.random.randn(a, b)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.random.randn(a, b)


def sin(a, cuda=False):
    if cuda:
        res = cp.sin(a)
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.sin(a)

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
        res = (target != cp.around(y)).mean()
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        y_one_hot = np.zeros(y.shape)
        i = 0
        for col in y:
            y_one_hot[i, np.argmax(col)] = 1
            i += 1
        return sklearn.metrics.zero_one_loss(np.around(y_one_hot), target)

def compute_random_fourier_features(num_params, input_train, target_train, input_test, target_test, relu=False, cuda=False, one_hot=False, unit_sphere=False):
    v = generate_random_fourier_matrix(num_params, unit_sphere, cuda)
    x = transform_inputs(v, input_train, relu, cuda)
    a = lstsq(x, target_train, cuda)
    preds = dot(x, a, cuda)
    mse_train = mse(preds, target_train, cuda)
    zero_one_train = zero_one(preds, target_train, cuda)
    x = transform_inputs(v, input_test, relu)
    preds = dot(x, a, cuda)
    mse_test = mse(preds, target_test)
    zero_one_test = zero_one(preds, target_test, cuda)
    a_norm = norm(a, cuda=cuda)

    return (mse_train, zero_one_train), (mse_test, zero_one_test), a_norm, a, v


def generate_random_fourier_matrix(num_params, unit_sphere, cuda):
    if unit_sphere:
        v = randn(num_params, 784, cuda)
        v /= norm(v, cuda=cuda)
    else:
        v = randn(num_params, 784, cuda) + 1/25
    return v


def transform_inputs(random_fourier_matrix, inputs, relu=False, cuda=False):
    transformed = dot(inputs, random_fourier_matrix.T, cuda)
    if relu:
        transformed = maximum(transformed, cuda)
    else:
        transformed = np.hstack([cos(transformed, cuda), sin(transformed, cuda)])
    return transformed


def mse(predictions, targets, cuda=False):
    if cuda:
        res = cp.square(cp.subtract(predictions, targets)).mean()
        cp.cuda.Stream.null.synchronize()
        return res
    else:
        return np.square(np.subtract(predictions, targets)).sum()/len(predictions)


def get_data(train=True, cuda=False, one_hot=False):
    subset_size = int(1e4)

    dataset = datasets.MNIST(root='./data',
                             download=True,
                             train=train,
                             transform=get_transform("mnist"))

    inputs = []
    targets = []

    for i in range(subset_size):
        inputs.append(np.array(dataset[i][0]))
        target = np.array(dataset[i][1])
        if one_hot:
            zeros = np.zeros(10)
            zeros[target] = 1
            target = zeros
        targets.append(target)

    inputs = np.array(inputs)
    targets = np.array(targets)
    if cuda:
        inputs = cp.asarray(inputs)
        targets = cp.asarray(targets)

    return inputs, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RFF or ReLu RFF using linear regression")
    parser.add_argument(
        "--config", type=str, dest="config", help="json file that holds config information", required=True)

    parser.add_argument("--relu", action="store_true",
                        help="Whether to use relu or not (default: False)", default=False)

    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use cuda or not (default: False)", default=False)

    parser.add_argument("--one_hot", action="store_true",
                        help="Whether to use one-hot encoding for output (default: False)", default=False)

    parser.add_argument("--unit_sphere", action="store_true",
                        help="Whether to sample vis from unit-sphere (will otherwise be sampled from N(0, 0.04)", default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    num_params = config["num_params"]["random_fourier"]
    relu = args.relu
    cuda = args.cuda
    unit_sphere = args.unit_sphere
    one_hot = args.one_hot
    mse_train_losses = []
    zero_one_train_losses = []
    mse_test_losses = []
    zero_one_test_losses = []
    norms = []

    input_train, target_train = get_data(cuda=cuda, one_hot=one_hot)
    input_test, target_test = get_data(train=False, cuda=cuda, one_hot=one_hot)

    if not relu:
        num_params = [p // 2 for p in num_params]
    for num in num_params:
        start = time.time()
        try:
            train_loss, test_loss, a_norm, features, transforms = compute_random_fourier_features(
            num, input_train, target_train, input_test, target_test, relu=relu, cuda=cuda, one_hot=one_hot, unit_sphere=unit_sphere)
            mse_train_losses.append(float(train_loss[0]))
            zero_one_train_losses.append(float(train_loss[1]))
            mse_test_losses.append(float(test_loss[0]))
            zero_one_test_losses.append(float(test_loss[1]))
            if one_hot:
                norms.append(a_norm.tolist())
            else:
                norms.append(float(a_norm))
            model_path = os.path.join("data", "models", "rff")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_name = str(num)
            if relu:
                model_name += "_relu"
            if cuda:
                model_name += "_cuda"
            if one_hot:
                model_name += "_one_hot"
            if unit_sphere:
                model_name += "_unit_sphere"
            features_fname = os.path.join(model_path, model_name + "_features")
            np.savez(features_fname, features)
            transforms_fname = os.path.join(model_path, model_name + "_transforms")
            np.savez(transforms_fname, transforms) 
        except MemoryError as e:
            print(e)
            break
        print("Model with {} params took {:d} mins and {:.4f}s".format(num, math.floor((time.time()-start)/60), (time.time()-start)%60))
        

    results = {}
    results["norms"] = norms
    results["num_parameters"] = num_params
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
    if one_hot:
        name += "_one_hot"
    if unit_sphere:
        name += "_unit_sphere"
    file_name = os.path.join(results_path, name + ".json")
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
