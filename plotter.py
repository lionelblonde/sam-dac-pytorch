from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import time

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

from helpers.math_util import smooth_out_w_ema
from helpers import logger


def plot(args, dest_dir, ycolkey, barplot):

    # font (must be first)
    font_dir = "/Users/lionelblonde/Library/Fonts/"
    if args.font == "Basier":
        f1 = fm.FontProperties(fname=Path(font_dir) / "BasierCircle-Regular.otf", size=20)
        f2 = fm.FontProperties(fname=Path(font_dir) / "BasierCircle-Regular.otf", size=32)
        f3 = fm.FontProperties(fname=Path(font_dir) / "BasierCircle-Regular.otf", size=22)
        f4 = fm.FontProperties(fname=Path(font_dir) / "BasierCircle-Medium.otf", size=24)
    elif args.font == "SourceCodePro":
        f1 = fm.FontProperties(fname=Path(font_dir) / "SourceCodePro-Light.otf", size=20)
        f2 = fm.FontProperties(fname=Path(font_dir) / "SourceCodePro-Regular.otf", size=32)
        f3 = fm.FontProperties(fname=Path(font_dir) / "SourceCodePro-Regular.otf", size=22)
        f4 = fm.FontProperties(fname=Path(font_dir) / "SourceCodePro-Medium.otf", size=24)
    else:
        raise ValueError("invalid font")

    marker_list = ["d", "X", "P", "*", "^", "s", "D", "v"]

    # palette
    palette = {
        "grid": (231, 234, 236),
        "face": (255, 255, 255),
        "axes": (200, 200, 208),
        "font": (108, 108, 126),
        "symbol": (64, 68, 82),
        "expert": (0, 0, 0),
        "curves": sns.color_palette(),
    }
    for k, v in palette.items():
        if k != "curves":
            palette[k] = tuple(float(e) / 255. for e in v)

    # figure color
    plt.rcParams["axes.facecolor"] = palette["face"]
    # dpi
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    # x and y axes
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.linewidth"] = 0.8
    # lines
    plt.rcParams["lines.linewidth"] = 1.4
    plt.rcParams["lines.markersize"] = 1
    # grid
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["grid.linestyle"] = "-"

    # dirs
    experiment_map = defaultdict(list)
    xcol_dump = defaultdict(list)
    ycol_dump = defaultdict(list)
    color_map = defaultdict(str)
    marker_map = defaultdict(str)
    text_map = defaultdict(str)
    dirs = [d.name for d in Path(args.dir).glob("*")]
    logger.info(f"pulling logs from sub-directories: {dirs}")
    dirs.sort()
    dnames = deepcopy(dirs)
    dirs = ["{}/{}".format(args.dir, d) for d in dirs]
    logger.info(dirs)

    # colors
    colors = {d: palette["curves"][i] for i, d in enumerate(dirs)}
    markers = {d: marker_list[i] for i, d in enumerate(dirs)}

    for d in dirs:

        path_glob = Path(d).glob("*/progress.csv")

        for fname in path_glob:
            # extract the expriment name from the file's full path
            experiment_name = fname.parent.parent.name
            # remove what comes after the uuid
            _i = 1 if args.round == 1 else 2  # directory naming has changed since (added git SHA)
            key = experiment_name.split(".")[0] + "." + experiment_name.split(".")[_i]
            env = experiment_name.split(".")[_i]
            experiment_map[env].append(key)
            # load data from the CSV file
            data = pd.read_csv(fname,
                               skipinitialspace=True,
                               usecols=[args.xcolkey, ycolkey])
            data.fillna(0.0, inplace=True)
            # retrieve the desired columns from the data
            xcol = data[args.xcolkey].to_numpy()
            ycol = data[ycolkey].to_numpy()
            # add the experiment"s data to the dictionary
            xcol_dump[key].append(xcol)
            ycol_dump[key].append(ycol)
            # add color
            color_map[key] = colors[d]
            # add marker
            marker_map[key] = markers[d]
            # add text
            text_map[key] = fname.parent.parent.parent.name

    for k, v in experiment_map.items():
        logger.info(k, v)

    # remove duplicate
    experiment_map = {k: list(set(v)) for k, v in experiment_map.items()}

    # display summary of the extracted data
    assert len(xcol_dump.keys()) == len(ycol_dump.keys())  # then use x col arbitrarily
    logger.info(f"summary -> {len(xcol_dump.keys())} different keys.")
    for i, key in enumerate(xcol_dump.keys()):
        logger.info(f"==== [key #{i}] {key} | #values: {len(xcol_dump[key])}")

    logger.info("\n>>>>>>>>>>>>>>>>>>>> Visualizing.")

    texts = deepcopy(dnames)
    texts.sort()
    texts = [text.split("__")[-1] for text in texts]
    logger.info(f"Legend texts (ordered): {texts}")

    patches = [plt.plot([],
                        [],
                        marker=marker_list[i],
                        ms=18,
                        ls="",
                        color=palette["curves"][i],
                        label="{:s}".format(texts[i]))[0]
               for i in range(len(texts))]

    # calculate the x axis upper bound
    xmaxes = defaultdict(int)
    for _, key in enumerate(xcol_dump.keys()):
        xmax = np.iinfo(np.int32).max  # max integer
        for _, key_ in enumerate(xcol_dump[key]):
            xmax = min(len(key_), xmax)
        logger.info(f"{key}: {xmax}")
        xmaxes[key] = xmax

    # Create constants from arguments to make the names more intuitive
    grid_size_x = args.grid_height
    grid_size_y = args.grid_width
    cell_size = 7
    fig, axs = plt.subplots(grid_size_x,
                            grid_size_y,
                            figsize=(cell_size * grid_size_y, cell_size * grid_size_x))

    if grid_size_x == 1:
        axs = np.expand_dims(axs, axis=0)
    if grid_size_y == 1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            axs[i, j].axis("off")

    # plot mean and standard deviation
    for j, env in enumerate(sorted(experiment_map.keys())):

        # create subplot
        ax = axs[j // grid_size_y, j % grid_size_y]
        ax.axis("on")

        # create grid
        ax.grid(color=palette["grid"])
        # only leave the left and bottom axes
        ax.spines["right"].set_visible(b=False)
        ax.spines["top"].set_visible(b=False)
        # set the color of the axes
        ax.spines["left"].set_color(palette["axes"])
        ax.spines["bottom"].set_color(palette["axes"])

        # only use for barplots but prevents pyright whining
        bars = {}
        bars_errors = {}
        bars_colors = {}

        if args.truncate >= 0:
            _xmaxes = [xmaxes[key] for key in experiment_map[env]]
            _xmax = np.amin(_xmaxes)
            for key in experiment_map[env]:
                xmaxes[key] = _xmax

        # go over the experiments and plot for each on the same subplot
        for _, key in enumerate(experiment_map[env]):

            xmax = deepcopy(xmaxes[key])

            logger.info(f"==== {key}, in color RGB={color_map[key]}")

            if len(ycol_dump[key]) > 1:
                # calculate statistics to plot
                mean = np.mean(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)
                std = np.std(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)

                # plot the computed statistics
                weight = 0.85
                smooth_mean = np.array(smooth_out_w_ema(mean, weight=weight))
                smooth_std = np.array(smooth_out_w_ema(std, weight=weight))

                if barplot:
                    bars[text_map[key]] = smooth_mean[-1]
                    bars_errors[text_map[key]] = smooth_std[-1]
                    bars_colors[text_map[key]] = color_map[key]
                else:
                    ax.plot(xcol_dump[key][0][0:xmax], smooth_mean,
                            marker=marker_map[key],
                            markersize=20,
                            markevery=args.markevery,
                            color=color_map[key],
                            alpha=1.0)
                    ax.fill_between(xcol_dump[key][0][0:xmax],
                                    smooth_mean - (args.stdfrac * smooth_std),
                                    smooth_mean + (args.stdfrac * smooth_std),
                                    facecolor=color_map[key],
                                    alpha=0.2)
            elif not barplot:
                ax.plot(xcol_dump[key][0], ycol_dump[key][0])
            else:
                pass

        if barplot:
            plot_name = False
            ax.bar(x=[(v.split("__")[-1] if plot_name else v.split("__")[0])
                      for v in sorted(set(text_map.values()))],
                   height=[bars[k] for k in sorted(bars.keys())],
                   yerr=[bars_errors[k] for k in sorted(bars_errors.keys())],
                   color=[bars_colors[k] for k in sorted(bars_colors.keys())],
                   width=0.6,
                   alpha=0.6,
                   capsize=5)
            for _, key in enumerate(experiment_map[env]):
                logger.info(key, text_map[key])
                _x = text_map[key].split("__")[-1] if plot_name else text_map[key].split("__")[0]
                ax.plot(_x, bars[text_map[key]],
                        marker=marker_map[key],
                        markersize=20,
                        color=color_map[key],
                        alpha=1.0)

        # create the axes labels
        ax.tick_params(width=0.2, length=1, pad=1,
                       colors=palette["axes"], labelcolor=palette["font"])
        if not barplot:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-4, 4),
                                useOffset=(False), useMathText=True)
        ax.xaxis.offsetText.set_fontproperties(f1)
        ax.xaxis.offsetText.set_position((0.95, 0))
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(f1)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(f1)
        if not barplot:
            ax.set_xlabel("Timesteps", color=palette["font"], fontproperties=f3)  # , labelpad=6
        ax.set_ylabel(args.ylabel, color=palette["font"], fontproperties=f3)  # , labelpad=12
        # create title
        ax.set_title(f"{env}", color=palette["font"], fontproperties=f4, pad=-10)

    # create legend
    legend = fig.legend(
        handles=patches,
        loc="center left",
        facecolor="w",
        bbox_to_anchor=(1.03, 0.5),
    )
    legend.get_frame().set_linewidth(0.0)
    for text in legend.get_texts():
        text.set_color(palette["font"])
        text.set_fontproperties(f2)

    fig.subplots_adjust(right=0.75)

    # save figure to disk
    plot_type_name = "barplot" if barplot else "plot"
    plt.savefig(f"{dest_dir}/plots_{ycolkey}_{plot_type_name}.pdf",
                format="pdf",
                bbox_inches="tight")
    logger.info("bye.")


if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser(description="Plotter")
    parser.add_argument("--font", type=str, default="SourceCodePro")
    parser.add_argument("--dir", type=str, default=None,
                        help="csv files location")
    parser.add_argument("--xcolkey", type=str, default=None,
                        help="name of the X column")
    parser.add_argument("--ycolkeys", nargs="+", type=str, default=None,
                        help="name of the Y column")
    parser.add_argument("--stdfrac", type=float, default=1.,
                        help="std envelope fraction")
    parser.add_argument("--round", type=int, default=2,
                        help="round logs were conducted at")
    parser.add_argument("--grid_width", type=int, default=3,
                        help="width of the grid in number of plots")
    parser.add_argument("--grid_height", type=int, default=3,
                        help="height of the grid in number of plots")
    parser.add_argument("--truncate", type=int, default=-1,
                        help="negative values prevent x truncation")
    parser.add_argument("--ylabel", type=str, default="Episodic Return",
                        help="Y-axis label")
    parser.add_argument("--markevery", type=int, default=124,
                        help="how often to put a mark")
    args = parser.parse_args()

    # create unique destination dir name
    hash_ = hashlib.sha1()
    hash_.update(str(time.time()).encode("utf-8"))
    dest_dir = f"plots/batchplots_{hash_.hexdigest()[:20]}"
    Path(dest_dir).mkdir(exist_ok=False)

    # plot
    for ycolkey in args.ycolkeys:
        plot(args, dest_dir=dest_dir, ycolkey=ycolkey, barplot=False)
        plot(args, dest_dir=dest_dir, ycolkey=ycolkey, barplot=True)
