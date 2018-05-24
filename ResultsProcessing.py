import pickle
import os
import numpy as np


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





