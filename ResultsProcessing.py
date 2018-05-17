import pickle
import os
import numpy as np


def defaulting_from_file(file):
    content = pickle.load(open(file, "rb"))
    n = content[0]["Defaulting"].shape[0]
    T = len(content)
    d = np.zeros((n, T))
    count = 0
    for i in range(0, T):
        d[:, count] = content[i]["Defaulting"]
    return d


def cdf_defaulting_from_file(file):
    content = pickle.load(open(file, "rb"))
    cdf_defaulting = np.cumsum(np.sum(content[0], axis=0))
    return cdf_defaulting


def cum_lost_from_file(file):
    content = pickle.load(open(file, "rb"))
    lost_value = np.cumsum(np.array(content[1]))
    return lost_value


def average_cdf_defaulting_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        cdf_defaulting = cdf_defaulting_from_file(folder + file)
        if first :
            avg = cdf_defaulting.copy()
            first = False
        else :
            avg += cdf_defaulting
        count += 1
    return (1 / count) * avg


def average_lost_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        print(file)
        lost = cum_lost_from_file(folder + file)
        if first:
            avg = lost.copy()
            first = False
        else:
            avg += lost
        count += 1
    return (1 / count) * avg


folder = "E:/Simulations/"


cdf_defaulting = average_cdf_defaulting_from_folder(folder)
cdf_lost = average_lost_from_folder(folder)




plt.plot(cdf_lost)

file = "E:/Simulations/ER0.5_Leverage10_2.pkl"

content = pickle.load(open(file, "rb"))

defaulting = content[0]
losses = content[1]

cdf_defaulting = np.cumsum(np.sum(defaulting, axis=0))