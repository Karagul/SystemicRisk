import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
font = {'family': 'normal', 'weight': 'bold', 'size': 22}
matplotlib.rc('font', **font)
plt.style.use('seaborn-deep')


def pickle_load(path):
    return pickle.load(open(path, "rb"))


def pickle_dump(path, obj):
    pickle.dump(obj, open(path, "wb"))


def aggregate_results(mc_results, i):
    s = len(mc_results)
    return np.array([mc_results[j][i] for j in range(0, s)])


def average_results(mc_results, i):
    arr = aggregate_results(mc_results, i)
    return arr.mean(axis=0)


def aggregate_results_sum(mc_results, i, k):
    resulti = aggregate_results(mc_results, i)
    resultk = aggregate_results(mc_results, k)
    return resulti + resultk


def average_results_sum(mc_results, i, k):
    arr = aggregate_results_sum(mc_results, i, k)
    return arr.mean(axis=0)




p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
lambda_star_grid = [1, 3, 5, 7, 10]
results_list = []
p_er = 0.1
for lamb in lambda_star_grid:
    results_list.append(pickle_load(
        "/home/dimitribouche/Bureau/Simulations/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl"))
    print(lamb)


cumdefaults_averages = [average_results(result, 1) for result in results_list]