import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_comparison(comparison_dict, ax=None):
    if not ax:
        fig, ax = plt.subplots(1)
    for key in comparison_dict.keys():
        ax.plot(comparison_dict[key], label=key)
        ax.legend()


def plot_init_comparison(comparison_dict, ax=None):
    if not ax:
        fig, ax = plt.subplots(1)
    for key in comparison_dict.keys():
        ax.hist(comparison_dict[key], label=key)
        ax.legend()


def plot_assets(prices, ax=None):
    if not ax:
        fig, ax = plt.subplots(1)
    m = prices.shape[1]
    for i in range(0, m):
        ax.plot(prices[:, i])


def plot_comparison_prices(comparison_dict,
                           prices,
                           suptitle="Comparison of graph structures",
                           x2="Time",
                           y1="Defaults CDF",
                           y2="Price"):
    fig, axes = plt.subplots(2)
    plot_comparison(comparison_dict, ax=axes[0])
    plot_assets(prices, ax=axes[1])
    plt.suptitle(suptitle)
    axes[0].set_ylabel(y1)
    axes[1].set_ylabel(y2)
    axes[1].set_xlabel(x2)


def plot_init_hists(comparison_dict, xlab="Equity"):
    fig, axes = plt.subplots(2, 2)
    keys_list = list(comparison_dict.keys())
    nsims = comparison_dict[keys_list[0]].shape[0]
    plt.suptitle("Comparisons of graphs structures for initial " + xlab)
    for i in [0, 1]:
        for j in [0, 1]:
            axes[i, j].hist(comparison_dict[keys_list[i*2 + j]])
            axes[i, j].set_title(keys_list[i*2 + j])
    axes[0, 0].set_ylabel("n draws (out of " + str(nsims) + ")")
    axes[1, 0].set_ylabel("n draws (out of " + str(nsims) + ")")
    axes[1, 0].set_xlabel(xlab)
    axes[1, 1].set_xlabel(xlab)
    plt.show()