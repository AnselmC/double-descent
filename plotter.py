import json
from collections import OrderedDict
import matplotlib.pyplot as plt

test_file = "data/results/epochs_600_cpu/two_layer_nn.json"
test_file = "data/results/two_layer_nn/CrossEntropyLoss/6000.json"
test_model = "two_layer_nn_3985"
def plot_double_descent(file_name):
    with open(file_name, "r") as fd:
        content = json.load(fd)

    sizes = sorted([int(x.split("_")[-1])/1e3 for x in content["Train losses"].keys()])
    train_losses = OrderedDict(sorted(content["Train losses"].items(), key= lambda t: int(t[0].split("_")[-1])))
    val_losses = OrderedDict(sorted(content["Val losses"].items(), key= lambda t: int(t[0].split("_")[-1])))
    final_train_losses = [v[-1] for v in train_losses.values()]
    final_val_losses = [v[-1] for v in val_losses.values()]
    test_losses = list(content["Test losses"].values())
    test_losses = list(OrderedDict(sorted(content["Test losses"].items(), key= lambda t: int(t[0].split("_")[-1]))).values())
    fig, ax = plt.subplots()
    ax.semilogx(sizes, final_train_losses, ".-", label="Train")
    ax.semilogx(sizes, final_val_losses, ".-", label="Validation")
    ax.semilogx(sizes, test_losses, ".-", label="Test")
    plt.axvline(40e3/1e3)
    fig.legend()
    plt.title("Double descent curve")
    plt.ylabel("Loss")
    plt.xlabel("Capacity (x1e3)")
    plt.show()
    
def plot_training(file_name):
    with open(file_name, "r") as fd:
        content = json.load(fd)

    for model_name in content["Train losses"].keys():
        plot_single_model(content, model_name)
    
def p():
    plot_double_descent(test_file)

def ps():
    plot_single_model(test_file, test_model)

def pt():
    plot_training(test_file)
    
def plot_single_model(content, model_name):
    train_losses = content["Train losses"][model_name]
    val_losses = content["Val losses"][model_name]
    fig, ax = plt.subplots()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title(model_name)
    plt.show()
    

