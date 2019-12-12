import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
import os
import argparse


def plot_results(results, file_prefix, label_color, background_color, transparent):
    num_params = np.array(results.pop("num_parameters"))
    if file_prefix == "reg":
        num_params *= 2
    norms = np.array(results.pop("norms"))
    titles = [x.replace("_", " ") for x in results.keys()]
    fig = plt.figure(figsize=(9, 6))
    ax1 = plt.subplot(221)
    ax1.set_xticklabels([])
    ax1.set_ylabel("Loss", color=label_color, size=9)
    ax2 = plt.subplot(222)
    ax2.set_ylabel("Loss in %", color=label_color, size=9)
    ax2.set_xticklabels([])
    ax3 = plt.subplot(223)
    ax3.set_ylabel("Loss", color=label_color, size=9)
    ax3.set_xlabel("N", color=label_color, size=9)
    ax4 = plt.subplot(224)
    ax4.set_ylabel("Loss in %", color=label_color, size=9)
    ax4.set_xlabel("N", color=label_color, size=9)
    axes = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes):
        key = list(results.keys())[i]
        data = np.array(results[key])
        if "zero" in titles[i]:
            data *= 100  # show as percentage
        if "test" in titles[i]:
            if file_prefix == "reg":
                if "zero" in titles[i]:
                    # TODO: load value from kernel machine result
                    ax.hlines(3.9, xmin=0, xmax=max(num_params),
                              color="C0", label="min. norm kernel")
                else:
                    # TODO: load value from kernel machine result
                    ax.hlines(0.13, xmin=0, xmax=max(num_params),
                              color="C0", label="min. norm kernel")

            ax.semilogy(num_params, data, ".-", color="orange", label="RFF" if file_prefix == "reg" else "RRF")
        else:
            ax.plot(num_params, data, ".-", color="orange", label="RFF" if file_prefix == "reg" else "RRF")
        ax.vlines(10000, ymin=0, ymax=max(data), color=label_color,
                  linestyle="dotted", label="interpolation")
        if "zero" in titles[i]:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.set_yticks([], minor=True)
        ax.set_title(titles[i].upper(), size=9, color=label_color)
        ax.set_facecolor(background_color)
        ax.tick_params(colors=label_color, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color(label_color)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(label_color)
        legend = ax.legend(frameon=False, fontsize=9)
        plt.setp(legend.get_texts(), color=label_color)
    fig.tight_layout()
    fig.set_facecolor(background_color)
    plt.savefig(file_prefix + "_losses.pdf", transparent=transparent)
    plt.show()

    norm_fig = plt.figure(figsize=(9/2, 6/2))
    ax = plt.subplot()

    try:
        ax.semilogy(num_params, norms.mean(axis=1), ".-", color="orange")
    except:
        ax.semilogy(num_params, norms, ".-", color="orange", label="RFF" if file_prefix == "reg" else "RRF")
    # TODO: load value from kernel machine result
    if file_prefix == "reg":
        ax.hlines(8, xmin=0, xmax=max(num_params),
                  color="C0", label="min. norm kernel")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(label_color)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.set_facecolor(background_color)
    ax.set_ylabel("L2 norm", color=label_color)
    ax.set_xlabel("N", color=label_color)
    ax.tick_params(colors=label_color, which="both")
    ax.set_yticks([], minor=True)
    legend = ax.legend(frameon=False)
    plt.setp(legend.get_texts(), color=label_color)
    norm_fig.set_facecolor(background_color)

    norm_fig.tight_layout()
    plt.savefig(file_prefix + "_norms.pdf", transparent=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot random feature results from experiments")

    parser.add_argument("results", type=argparse.FileType('r'),
                        nargs='+', help="the result json file(s). Expected in order of increasing capacity sizes")

    parser.add_argument("--saveto", dest="file_prefix", type=str,
                        help="The file prefix to save the plots to", required=True)

    parser.add_argument("--bg", dest="bg", type=str,
                        help="The desired background color of the plot. Any acceptable matplotlib string (default \"black\")", default="black")
    parser.add_argument("--fg", dest="fg", type=str,
                        help="The desired label color of the plot. Any acceptable matplotlib string (default \"white\")", default="white")
    parser.add_argument("--transparent", action="store_true",
                        help="Save plot with transparent background", default=False)

    args = parser.parse_args()
    transparent = args.transparent
    bg = args.bg
    fg = args.fg
    results = {}
    for result in args.results:
        result = json.load(result)
        for k, v in result.items():
            if k in results:
                results[k] += v
            else:
                results[k] = v

    plot_results(results, args.file_prefix, fg, bg, transparent)
