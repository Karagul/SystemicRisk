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

