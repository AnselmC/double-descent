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
TODO
### Two layer Neural Network
TODO
## Contributing

If you would like to contribute to this venture (by adding new models, datasets, etc.) please feel free to create a pull request.
