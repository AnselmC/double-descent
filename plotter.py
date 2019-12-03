import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

test_file = "data/results/two_layer_nn/CrossEntropyLoss/1000.json"
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
    ax.set_facecolor("black")
    ax.semilogx(sizes, final_train_losses, ".-", label="Train", color="C0")
    ax.semilogx(sizes, test_losses, ".-", label="Test", color="C1")
    ax.set_xticks([], minor=True)
    ax.set_xticks([10, 40, 100, 300, 800], minor=False)
    ax.vlines(40e3/1e3, ymin=0, ymax=max(test_losses), linestyle="dotted" , color="white", label="Interpolation")
    ax.tick_params(colors="white", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_color("white")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    legend = ax.legend(frameon=False, fontsize=9)
    plt.setp(legend.get_texts(), color="w")
    plt.ylabel("Loss", color="white")
    plt.xlabel(r"N $(\times 10^3)$", color="white")
    plt.savefig("../presentation/resources/two_layer_nn.pdf", transparent=True)
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
    
if __name__=="__main__":
    p()
