import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_comparison(comparison_dict, ax=None):
    if not ax:
        fig, ax = plt.subplots(1)
    for key in comparison_dict.keys():
        ax.plot(comparison_dict[key], label=key)
        ax.legend()
    plt.show()


def plot_assets(prices, ax=None):
    if not ax:
        fig, ax = plt.subplots(1)
    m = prices.shape[0]
    for i in range(0, m):
        ax.plot(prices)
    plt.show()


def plot_comparison_prices(comparison_dict, prices):
    fig, axes = plt.subplots(2)
    plot_comparison(comparison_dict, ax=axes[0])
    plot_assets(prices, ax=axes[1])
    plt.show()