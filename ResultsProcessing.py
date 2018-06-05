import pickle
import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 20,
                            'font.weight': "medium",
                            'figure.titleweight': "semibold",
                            'lines.linewidth': 3,
                            'axes.linewidth': 1.2})

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


def safe_conversion(x):
    if x - int(x) == 0:
        return str(int(x))
    else:
        return str(float(x))


def indicator_after_tperiods(indicator_dict, t):
    per_grid = [float(k[0]) for k in indicator_dict.keys()]
    per_grid.sort()
    lamb_grid = [float(k[1]) for k in indicator_dict.keys()]
    lamb_grid.sort()
    print(per_grid)
    print(lamb_grid)
    X, Y = np.meshgrid(per_grid, lamb_grid)
    Z = np.zeros(X.shape)
    for j in range(0, len(per_grid)):
        for i in range(0, len(lamb_grid)):
            Z[i, j] = indicator_dict[str(X[i, j]), safe_conversion(Y[i, j])][t]
    return X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)


def threshold_time(probas, thresh):
    try:
        return min(np.argwhere(probas > thresh))
    except ValueError:
        return probas.shape[0]


def systemic_threshold_time(probas_dict, thresh):
    systemic_dict = dict()
    per_grid = [float(k[0]) for k in probas_dict.keys()]
    per_grid.sort()
    lamb_grid = [float(k[1]) for k in probas_dict.keys()]
    lamb_grid.sort()
    for per in per_grid:
        for lamb in lamb_grid:
            systemic_dict[str(per), str(lamb)] = threshold_time(probas_dict[str(per), str(lamb)], thresh)
    return systemic_dict





p_ers_grid = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_star_grid = [2, 3, 4, 5, 7.5, 10]
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# p_ers_bis = p_ers_grid
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
cumdefaults_dict = dict()
lost_dict = dict()
probas_dict = dict()
degrees_dict = dict()
for p_er in p_ers_grid:
    for lamb in lambda_star_grid:
        # file = pickle_load("/home/dimitribouche/Bureau/Simulations/FullRun/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
        file = pickle_load(
            "/home/dimitribouche/Bureau/Simulations/Rewiring/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
        # file = pickle_load(
        #     "/home/dimitribouche/Bureau/TestsSimus/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
        cumdefaults_dict[(str(p_er), str(lamb))] = average_results(file, 1)
        lost_dict[(str(p_er), str(lamb))] = average_results(file, 2)
        probas_dict[(str(p_er), str(lamb))] = empirical_proba_infto(file, 1, 25)
        degrees_dict[(str(p_er), str(lamb))] = average_results_sum(file, 3, 4)
        print(lamb)
    print(p_er)


# pickle_dump("/home/dimitribouche/Bureau/Simulations/Computations/proba25_dict.pkl", probas_dict)
# pickle_dump("/home/dimitribouche/Bureau/Simulations/Computations/lost_dict.pkl", lost_dict)
# pickle_dump("/home/dimitribouche/Bureau/Simulations/Computations/degrees_dict.pkl", degrees_dict)


pickle_dump("/home/dimitribouche/Bureau/Simulations/Rewiring/Computations/cumdefaults_dict.pkl", cumdefaults_dict)
pickle_dump("/home/dimitribouche/Bureau/Simulations/Rewiring/Computations/proba25_dict.pkl", probas_dict)
pickle_dump("/home/dimitribouche/Bureau/Simulations/Rewiring/Computations/lost_dict.pkl", lost_dict)
pickle_dump("/home/dimitribouche/Bureau/Simulations/Rewiring/Computations/degrees_dict.pkl", degrees_dict)


cumdefaults_dict = pickle_load("/home/dimitribouche/Bureau/Simulations/Computations/cumdefaults_dict.pkl")

p_ers_bis = [0.01, 0.025, 0.05, 0.075, 0.1, 0.4, 0.7, 1.0]
plt.figure()
lamb = 5
for p_er in p_ers_bis[::-1]:
    plt.plot(cumdefaults_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    plt.legend()
plt.subplots_adjust(top=0.92,
                    bottom=0.095,
                    left=0.075,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)
plt.xlabel("Period")
plt.ylabel("Cumulative defaults (out of 100)")
plt.show()


p_er = 0.01
plt.figure()
for lamb in lambda_star_grid[::-1]:
    plt.plot(cumdefaults_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    plt.legend()
plt.subplots_adjust(top=0.92,
                    bottom=0.095,
                    left=0.075,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)
plt.xlabel("Period")
plt.ylabel("Cumulative defaults (out of 100)")
plt.show()




probas_dict = pickle_load("/home/dimitribouche/Bureau/Simulations/Computations/proba25_dict.pkl")

# p_ers_bis = [0.01, 0.025, 0.05, 0.075, 0.1, 0.4, 0.7, 1.0]
plt.figure()
lamb = 10
for p_er in p_ers_bis[::-1]:
    plt.plot(probas_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    plt.legend()
plt.subplots_adjust(top=0.92,
                    bottom=0.095,
                    left=0.075,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)
plt.xlabel("Period")
plt.ylabel("Probability of 25% of defaults")
plt.show()



lost_dict = pickle_load("/home/dimitribouche/Bureau/Simulations/Computations/lost_dict.pkl")

# p_ers_bis = [0.01, 0.025, 0.05, 0.075, 0.1, 0.4, 0.7, 1.0]
plt.figure()
lamb = 5
for p_er in p_ers_bis[::-1]:
    plt.plot(lost_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    plt.legend()
plt.subplots_adjust(top=0.92,
                    bottom=0.095,
                    left=0.075,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)
plt.xlabel("Period")
plt.ylabel("Fraction of initial value lost")
plt.show()



degrees_dict = pickle_load("/home/dimitribouche/Bureau/Simulations/Liquidator/Computations/degrees_dict.pkl")

#p_ers_bis = [0.01, 0.025, 0.05, 0.075, 0.1, 0.4, 0.7, 1.0]
plt.figure()
lamb = 10
for p_er in p_ers_grid[::-1]:
    plt.plot(0.01 * degrees_dict[(str(p_er), str(lamb))], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    plt.legend()
plt.subplots_adjust(top=0.92,
                    bottom=0.095,
                    left=0.075,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)
plt.xlabel("Period")
plt.ylabel("Average degree")
plt.show()



######################3 D ##############################"""

systemic_dict = systemic_threshold_time(probas_dict, 0.25)



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y, Z = indicator_after_tperiods(lost_dict, 400)


# # Plot the surface.
surf = ax.plot_wireframe(X, Y, Z)
ax.set_xlabel("parameter of ER graph")
ax.set_ylabel("Leverage")
ax.set_zlabel("Fraction of initial value lost at T=400")

ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
ax.zaxis.tickpad = 30

plt.subplots_adjust(top=0.975,
                    bottom=0.065,
                    left=0.045,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)









p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# p_ers_bis = p_ers_grid
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
cumdefaults_dict = dict()
lost_dict = dict()
for p_er in p_ers_grid:
    # file = pickle_load("/home/dimitribouche/Bureau/Simulations/FullRun/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
    file = pickle_load("/media/dimitribouche/LaCie/SimulationsPrice/Rewiring/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
    cumdefaults_dict[str(p_er)] = average_results(file, 1)
    lost_dict[str(p_er)] = average_results(file, 2)
    print(p_er)


p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# p_ers_bis = p_ers_grid
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
cumdefaults_dict_liq = dict()
lost_dict_liq = dict()
for p_er in p_ers_grid:
    # file = pickle_load("/home/dimitribouche/Bureau/Simulations/FullRun/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
    file = pickle_load("/media/dimitribouche/LaCie/SimulationsPrice/Liquidator/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl")
    cumdefaults_dict_liq[str(p_er)] = average_results(file, 1)
    lost_dict_liq[str(p_er)] = average_results(file, 2)
    print(p_er)




prices = pickle_load("/media/dimitribouche/LaCie/SimulationsPrice/Prices.pkl")



fig, axes = plt.subplots(3, sharex=True)
axes[2].plot(prices, c="#95a5a6")
lamb = 5
for p_er in p_ers_grid[::-1]:
    axes[0].plot(cumdefaults_dict[str(p_er)], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
    axes[1].plot(lost_dict[str(p_er)], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
axes[0].legend()
axes[1].legend()
axes[2].set_xlabel("Period")
axes[2].set_ylabel("Prices")
axes[0].set_ylabel("Fraction of initial value lost")
axes[1].set_ylabel("Cumulative defaults (out of 100)")




fig, axes = plt.subplots(2, sharex=True)
axes[1].plot(prices, c="#95a5a6")
lamb = 5
for p_er in p_ers_grid[::-1]:
    axes[0].plot(lost_dict_liq[str(p_er)], label="$\lambda^{\star}$=" + str(lamb) + "; p=" + str(p_er))
axes[0].legend()
axes[1].set_xlabel("Period")
axes[1].set_ylabel("Prices")
axes[0].set_ylabel("Fraction of initial value lost")

