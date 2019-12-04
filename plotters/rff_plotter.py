import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
import os
import argparse


def plot_results(results, file_prefix):
    num_params = 2* np.array(results.pop("num_parameters"))
    norms = np.array(results.pop("norms"))
    titles = [x.replace("_", " ") for x in results.keys()]
    fig = plt.figure(figsize=(9,6))
    ax1 = plt.subplot(221)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Loss", color="white", size=9)
    ax2 = plt.subplot(222)
    ax2.set_ylabel("Loss in %", color="white", size=9)
    ax2.set_xticklabels([])
    ax3 = plt.subplot(223)
    ax3.set_ylabel("Loss", color="white", size=9)
    ax3.set_xlabel("N", color="white", size=9)
    ax4 = plt.subplot(224)
    ax4.set_ylabel("Loss in %", color="white", size=9)
    ax4.set_xlabel("N", color="white", size=9)
    axes = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes):
        key = list(results.keys())[i]
        data = np.array(results[key])
        if "zero" in titles[i]:
            data *= 100 # show as percentage
        if "test" in titles[i]:
            if "zero" in titles[i]:
                ax.hlines(3.9, xmin=0, xmax=max(num_params), color="C0", label="min. norm kernel")
            else:
                ax.hlines(0.13, xmin=0, xmax=max(num_params), color="C0", label="min. norm kernel")
            
            ax.semilogy(num_params, data, ".-", color="orange", label="RFF")
        else:
            ax.plot(num_params, data, ".-", color="orange", label="RFF")
        ax.vlines(10000, ymin=0, ymax=max(data), color="white", linestyle="dotted", label="interpolation")
        if "zero" in titles[i]:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.set_yticks([], minor=True)
        ax.set_title(titles[i].upper(), size=9, color="white")
        ax.set_facecolor("black")
        ax.tick_params(colors="white", which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.spines["right"].set_visible(False) 
        ax.spines["left"].set_color("white")
        legend = ax.legend(frameon=False, fontsize=9)
        plt.setp(legend.get_texts(), color="w")
    fig.tight_layout()
    fig.set_facecolor("black")
    plt.savefig(file_prefix + "_losses.pdf", transparent=True)
    plt.show()

    norm_fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot()

    try:
        ax.semilogy(num_params, norms.mean(axis=1), ".-", color="orange")
    except:
        ax.semilogy(num_params, norms, ".-", color="orange", label="RFF")
    ax.hlines(8, xmin=0, xmax=max(num_params), color="C0", label="min. norm kernel")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_color("white")
    ax.set_facecolor("black")
    ax.set_ylabel("L2 norm", color="white")
    ax.set_xlabel("N", color="white")
    ax.tick_params(colors="white", which="both")
    ax.set_yticks([], minor=True)
    legend = ax.legend(frameon=False)
    plt.setp(legend.get_texts(), color="w")
    norm_fig.set_facecolor("black")

    norm_fig.tight_layout()
    plt.savefig(file_prefix + "_norms.pdf", transparent=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results from  experiments")

    parser.add_argument("results", type=argparse.FileType('r'),
                        nargs='+', help="the result json file(s).")

    parser.add_argument("--saveto, -s", dest="file_prefix", type=str, help="The file prefix to save the plots to", required=True)
    args = parser.parse_args()
    results = {}
    for result in args.results:
        result = json.load(result)
        for k, v in result.items():
            if k in results:
                results[k] += v
            else:
                results[k] = v


    plot_results(results, args.file_prefix)
