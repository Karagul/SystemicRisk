import pickle
import os
import numpy as np


def list_averaging(list_of_lists):
    first = True
    count = 0
    for l in list_of_lists:
        ar = np.array(l)
        if first:
            avg = ar.copy()
            first = False
        else:
            avg += ar
    avg *= (1 / count)
    return avg


def aggregate_results(mc_results, i):
    s = len(mc_results)
    return [mc_results[j][i] for j in range(0, s)]


