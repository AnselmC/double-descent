# The double-descent curve

This repository was created as part of the seminar _Generalization and Optimization in Deep Learning_ led by Thomas Frerix
during the winter semester 2019/2020 at the TU Munich.

The goal is to reproduce the results of the paper [_Reconciling modern machine learning practices with the classical bias-variance tradeoff_](https://www.researchgate.net/publication/334663035_Reconciling_modern_machine-learning_practice_and_the_classical_bias-variance_trade-off) by Mikhail et al at Ohio State.

Currently, it is possible to compute the random feature models (both fourier and relu) and the two layer net on the MNIST dataset.


## Usage
### Getting started
To get started you can install _numpy_, _matplotlib_, and _sklearn_ easily using the `requirements.txt`:
```bash
pip install -r requirements.txt
```

This will allow you to compute the Random Feature models on your CPU and plot the results.
If you want to use your GPU, you will need to install _cupy_: `pip install cupy`.
If you want to compute the two layer net, you'll need _pytorch_ and install the correct version from [here](https://pytorch.org/get-started/locally/)

### Random features
#### Training
```bash
usage: random_features.py [-h] --config CONFIG [--relu] [--cuda] [--one_hot]
                          [--unit_sphere]

Compute RFF or ReLu RFF using linear regression

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  json file that holds config information
  --relu           Whether to use relu or not (default: False)
  --cuda           Whether to use cuda or not (default: False)
  --one_hot        Whether to use one-hot encoding for output (default: False)
  --unit_sphere    Whether to sample vis from unit-sphere (will otherwise be
                   sampled from N(0, 0.04)
```

#### Plotting
```bash
usage: random_feature_plotter.py [-h] --saveto FILE_PREFIX [--bg BG] [--fg FG]
                                 [--transparent]
                                 results [results ...]

Plot random feature results from experiments

positional arguments:
  results               the result json file(s). Expected in order of
                        increasing capacity sizes

optional arguments:
  -h, --help            show this help message and exit
  --saveto FILE_PREFIX  The file prefix to save the plots to
  --bg BG               The desired background color of the plot. Any
                        acceptable matplotlib string (default "black")
  --fg FG               The desired label color of the plot. Any acceptable
                        matplotlib string (default "white")
  --transparent         Save plot with transparent background
```

### Two layer Neural Network
#### Training
```bash
usage: train_nn.py [-h] [--model {0,1,2}] [--prev PREV] [--num NUM] --config
                   CONFIG [--loss {0,1,2}] [--epochs EPOCHS]

Run multiple trainings with growing network capacity

optional arguments:
  -h, --help       show this help message and exit
  --model {0,1,2}  select model to train: 0 [two_layer_net], 1 [Random Fourier
                   Features], 2 [Random ReLU Features]
  --prev PREV      run single model size given path to model of previous size
  --num NUM        the number of model sizes to run, starting at smallest
                   model or model given with --prev flag
  --config CONFIG  config file for training
  --loss {0,1,2}   The loss function to use 0 [Squared](default), 1 [Zero-
                   one], 2 [Cross-Entropy]
  --epochs EPOCHS  The number of epochs to train, if not given, the value in
                   the config file is taken
```
### Plotting
```bash
usage: random_feature_plotter.py [-h] --saveto FILE_PREFIX [--bg BG] [--fg FG]
                                 [--transparent]
                                 results [results ...]

Plot random feature results from experiments

positional arguments:
  results               the result json file(s). Expected in order of
                        increasing capacity sizes

optional arguments:
  -h, --help            show this help message and exit
  --saveto FILE_PREFIX  The file prefix to save the plots to
  --bg BG               The desired background color of the plot. Any
                        acceptable matplotlib string (default "black")
  --fg FG               The desired label color of the plot. Any acceptable
                        matplotlib string (default "white")
  --transparent         Save plot with transparent background
```

## Contributing

If you would like to contribute to this venture (by fixing bugs, adding features, models, datasets, etc.) please feel free to create a pull request.
