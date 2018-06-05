import pickle
import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 40,
                            'font.weight': "semibold",
                            'figure.titleweight': "semibold",
                            'lines.linewidth': 5,
                            'axes.linewidth': 3})

import matplotlib.pyplot as plt


def pickle_load(path):
    return pickle.load(open(path, "rb"))


def pickle_dump(path, obj):
    pickle.dump(obj, open(path, "wb"))


def aggregate_results(mc_results, i):
    s = len(mc_results)
    return np.array([mc_results[j][i] for j in range(0, s)]).astype(np.float64)


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


def empirical_proba_infto(mc_results, i, thresh):
    aggr = aggregate_results(mc_results, i)
    aggr_ind = (aggr > thresh).astype(int)
    return aggr_ind.mean(axis=0)



#
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [1, 3, 5, 7, 10]
# cumdefaults_dict = dict()
# lost_dict = dict()
# probas_dict = dict()
# for p_er in p_ers_grid:
#     for lamb in lambda_star_grid:
#         file = pickle_load("E:/Simulations/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
#         # cumdefaults_dict[(str(p_er), str(lamb))] = average_results(file, 1)
#         # lost_dict[(str(p_er), str(lamb))] = average_results(file, 2)
#         probas_dict[(str(p_er), str(lamb))] = empirical_proba_infto(file, 1, 25)
#         print(lamb)
#     print(p_er)
#
#
# plt.figure()
# lamb = 5
# for p_er in p_ers_grid:
#     plt.plot(probas_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
#     plt.legend()
# plt.show()
#
#
#
#
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)