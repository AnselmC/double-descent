import json
import matplotlib.pyplot as plt

test_file = "data/results/epochs_600/two_layer_nn.json"
test_model = "two_layer_nn_3985"
def plot_double_descent(file_name):
    with open(file_name, "r") as fd:
        content = json.load(fd)

    sizes = [int(x.split("_")[-1])/1e3 for x in content["Train losses"].keys()]
    final_train_losses = [v[-1] for v in content["Train losses"].values()]
    final_val_losses = [v[-1] for v in content["Val losses"].values()]
    test_losses = list(content["Test losses"].values())
    fig, ax = plt.subplots()
    ax.semilogx(sizes, final_train_losses, ".-", label="Train")
    ax.semilogx(sizes, final_val_losses, ".-", label="Validation")
    ax.semilogx(sizes, test_losses, ".-", label="Test")
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
    

