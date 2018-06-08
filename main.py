# Third party modules
import numpy as np
import networkx as nx
import BankNetwork
import RiskyAssets
import importlib
import time
import pickle
import pathlib
import matplotlib.pyplot as plt

# Local imports
import BalanceSheetInit as BSI
import GraphInit as GI
import InitAnalysis as InitA
import SimulationsTools as ST
import Measures
import Visualization as Viz
# import ResultsProcessing as RP


importlib.reload(ST)
importlib.reload(Measures)
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(InitA)
importlib.reload(Viz)
importlib.reload(BSI)


def pickle_dump(path, obj):
    pickle.dump(obj, open(path, "wb"))


def pickle_load(path):
    return pickle.load(open(path, "rb"))




#### Dictionary of parameters
params = dict()


#### Fundamental parameters
n = 100
m = 2
T = 1000
params["m"] = m
params["T"] = T
params["n"] = n
# params["r"] = 0.02
r_annual = 0.05
params["r"] = ST.daily_compound(r_annual, 365)
params["xi"] = 0.7
params["zeta"] = 0.7
params["theta"] = - np.log(0.2)
params["lambda_star"] = 5


### RISKY ASSETS PARAMETERS
x0 = 10
mu = 0.01
sigma = 0.3
init_val = x0 * np.ones((m, ))
# mus = mu * np.ones((m, ))
mus = np.array([0.01, 0.01])
# sigmas = sigma * np.ones((m, ))
sigmas = np.array([0.3, 0.3])
assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
prices = assets.generate()
plt.figure()
for i in range(0, m):
    plt.plot(prices[:, i])

### CHOICE OF RISKY ASSETS
# Max diversification
# ws = (1 / m) * np.ones((m, ))
# Min diversification
ws = np.zeros((m, ))
for i in range(0, m//2):
    ws[i] = 1 / (m // 2)
# ws[0] = 1
qinit = BSI.QInit(n, ws)
params["q"] = qinit.random_asset_choice()


### BALANCE SHEET INITIALIZATION PARAMETERS
params["liquidator"] = True
params["enforce_leverage"] = False
# Nominal value of all loans (and debts)
l = 1000
params["l"] = l
# Minimal equity for all banks
params["e"] = 10000
# Value of alpha parameter for all banks
alpha = 0.25
# Value of beta parameter for all banks
beta = 1
# Value of \bar E parameter for all banks
bar_e = 5000
params["alphas"] = alpha * np.ones((n, ))
params["betas"] = beta * np.ones((n, ))
params["bar_E"] = bar_e * np.ones((n, ))



#
# # On average 1 edge out of 2 is negative and 1 out of 2 is positive
p_sign = 0.5
# ER parameter
p_er = 0.5
# Values of loans and their respective probabilities
# vals = np.array([2 * l / p_er])
vals = np.array([l])
distrib = np.array([1])
# Graph structure
#graph = nx.cycle_graph(n)
graph = nx.erdos_renyi_graph(n, p_er)
# graph = nx.complete_graph(n)
graph = GI.GraphInit(graph)
# Number of Monte Carlo iterations for
n_mc_graph = 10
# MC on random allocations on graph
start = time.time()
# mc_list, lev = ST.mc_on_graphs(params, prices, x0, mus, graph, n_mc_graph, p_sign, vals, distrib)
# results = ST.iterate_periods(params, prices, x0, mus, graph, p_sign, vals, distrib)
mc_graph = ST.mc_on_er_graphs(params, prices, x0, mus, p_er, n_mc_graph, p_sign, vals, distrib)
end = time.time()
print(end - start)


p_sign = 0.5
distrib = np.array([1])
vals = np.array([l])
p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [1, 3, 5, 7, 10]
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
# p_ers_grid = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambda_star_grid = [2, 3, 4, 5, 7.5, 10]
n_mc_graph = 1000

start = time.time()
for p_er in p_ers_grid:
    save_out = "/home/dimitribouche/Bureau/SimulationsShort/Liquidator/p_er=" + str(p_er) + "_leverage=" + str(params["lambda_star"]) + ".pkl"
    # save_out = "/home/dimitribouche/Bureau/TestsSimus/p_er=" + str(p_er) + "_leverage=" + str(
    #     lamb) + ".pkl"
    # results = ST.mc_full_er(params, prices_list, x0, mus, p_er, p_sign, vals, distrib)
    results = ST.mc_on_er_graphs(params, prices, x0, mus, p_er, n_mc_graph, p_sign, vals, distrib)
    pickle_dump(save_out, results)
    print("p_er :" + str(p_er))
end = time.time()

print(end - start)






p_sign = 0.5
distrib = np.array([1])
vals = np.array([l])
p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [1, 3, 5, 7, 10]
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
# p_ers_grid = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambda_star_grid = [2, 3, 4, 5, 7.5, 10]
n_mc_prices = 100
prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)

start = time.time()
for p_er in p_ers_grid:
    save_out = "/home/dimitribouche/Bureau/SimulationsShort/Rewiring/p_er=" + str(p_er) + "_leverage=" + str(params["lambda_star"]) + ".pkl"
    # save_out = "/home/dimitribouche/Bureau/TestsSimus/p_er=" + str(p_er) + "_leverage=" + str(
    #     lamb) + ".pkl"
    results = ST.mc_full_er(params, prices_list, x0, mus, p_er, p_sign, vals, distrib)
    pickle_dump(save_out, results)
    print("p_er :" + str(p_er))
end = time.time()

print(end - start)













p_sign = 0.5
distrib = np.array([1])
vals = np.array([l])
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [1, 3, 5, 7, 10]
# p_ers_grid = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
# lambda_star_grid = [3, 5, 7.5, 10]
# p_ers_grid = [0.6, 1.0]
# lambda_star_grid = [5, 10]
p_ers_grid = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_star_grid = [2, 3, 4, 5, 7.5, 10]
n_mc_prices = 4000
prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)

start = time.time()
for p_er in p_ers_grid:
    for lamb in lambda_star_grid:
        save_out = "/home/dimitribouche/Bureau/Simulations/Rewiring/p_er=" + str(p_er) + "_leverage=" + str(lamb) + ".pkl"
        # save_out = "/home/dimitribouche/Bureau/TestsSimus/p_er=" + str(p_er) + "_leverage=" + str(
        #     lamb) + ".pkl"
        params["lambda_star"] = lamb
        results = ST.mc_full_er(params, prices_list, x0, mus, p_er, p_sign, vals, distrib)
        pickle_dump(save_out, results)
        print("lambda : " + str(lamb))
        print("p_er :" + str(p_er))
end = time.time()

print(end - start)






#
# # On average 1 edge out of 2 is negative and 1 out of 2 is positive
# p_sign = 0.5
# # ER parameter
# p_er = 0.5
# # Values of loans and their respective probabilities
# vals = np.array([2 * l / p_er])
# distrib = np.array([1])
# # Graph structure
# lambda_star_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# results_list = []
#
# n_mc_prices = 100
# prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)
#
# start = time.time()
# for lamb in lambda_star_grid:
#     params["lambda_star"] = lamb
#     #graph = nx.cycle_graph(n)
#     graph = nx.erdos_renyi_graph(n, p_er)
#     # graph = nx.complete_graph(n)
#     graph = GI.GraphInit(graph)
#     # MC on random allocations on graph
#     # mc_list, lev = ST.mc_on_graphs(params, prices, x0, mus, graph, n_mc_graph, p_sign, vals, distrib)
#     results_list.append(ST.mc_on_prices(params, prices_list, x0, mus, graph, p_sign, vals, distrib))
#     print(lamb)
# end = time.time()
# print(end - start)
#
# avg_list = [RP.average_results(mc_result, 1) for mc_result in results_list]
#
# plt.figure()
# for i in range(0, 9):
#     plt.plot(avg_list[i], label=lambda_star_grid[i])
# plt.legend()
#
#

# ### MC on prices
# # On average 1 edge out of 2 is negative and 1 out of 2 is positive
# p = 0.5
#
# #er_ps = [0.005, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# er_ps = [0.06, 0.07, 0.08, 0.09]
# er_ps = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25,
#          0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
#
# # er_ps = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# for p_er in er_ps :
#
# # Values of loans and their respective probabilities
#     vals = np.array([2*l/p_er])
#     distrib = np.array([1])
# # Graph structure
# #graph = nx.cycle_graph(n)
#     graph = nx.erdos_renyi_graph(n, p_er)
# # graph = nx.complete_graph(n)
#     graph = GI.GraphInit(graph)
# # Number of Monte Carlo iterations for
#     n_mc_graph = 1
#     n_mc_prices = 1000
#     prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)
#     path = "/home/dimitribouche/Bureau/Simulations/ER" + str(p_er) + "_Leverage" + str(params["lambda_star"]) + "/"
#     start = time.clock()
# #    mc_dict = ST.mc_on_prices_ergraphs(params, prices_list, x0, mus, er_ps, n_mc_graph, p, vals, distrib)
#     mc_prices = ST.mc_on_prices(params, prices_list, x0, mus, graph, n_mc_graph, p, vals, distrib, save_out=path)
#     end = time.clock()
#     print(end - start)
#
#
#
#



# p_er=0.5
# vals = np.array([2*l/p_er])
# distrib = np.array([1])
# # Graph structure
# #graph = nx.cycle_graph(n)
# graph = nx.erdos_renyi_graph(n, p_er)
# # graph = nx.complete_graph(n)
# graph = GI.GraphInit(graph)
# # Number of Monte Carlo iterations for
# n_mc_graph = 1
# n_mc_prices = 1000
# prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)
# path = "/media/dimitribouche/Elements/Simulations/ER" + str(p_er) + "_Leverage" + str(params["lambda_star"]) + "/"
# start = time.clock()
# #    mc_dict = ST.mc_on_prices_ergraphs(params, prices_list, x0, mus, er_ps, n_mc_graph, p, vals, distrib)
# mc_prices = ST.mc_on_prices(params, prices_list, x0, mus, graph, n_mc_graph, p, vals, distrib, save_out=path)
# end = time.clock()
# print(end - start)












# Comparison of several Erdos Reyni graphs
# er_params = [0.01, 0.05, 0.2, 0.5, 0.8, 1]
# er_comparisons = ST.compare_ER_graphs(params, prices, x0, mus, er_params, n_mc, p, vals, distrib)
# er_defaults_cdf = Measures.average_defaults_cdf_dict(er_comparisons)
# Viz.plot_comparison_prices(er_defaults_cdf,
#                            prices,
#                            suptitle="ER graphs, differents ps - Minimum asset diversification - lambda_star="
#                                     + str(params["lambda_star"])
#                                     + "- 100 banks - 10 simus per graph")
#

# Comparison of several graph structures
# graphs_dict = dict()
# cycle_graph = nx.cycle_graph(n)
# complete_graph = nx.complete_graph(n)
# star_graph = nx.star_graph(n-1)
# er_graph = nx.erdos_renyi_graph(n, 0.05)
# graphs_dict["Circle"] = GI.GraphInit(cycle_graph)
# graphs_dict["Star"] = GI.GraphInit(star_graph)
# graphs_dict["ER - 0.05"] = GI.GraphInit(er_graph)
# graphs_dict["Complete"] = GI.GraphInit(complete_graph)
# graph_comparisons = ST.compare_graphs(params, prices, x0, mus, graphs_dict, n_mc, p, vals, distrib)
# graph_comparisons_defaults = Measures.average_defaults_cdf_dict(graph_comparisons)
# Viz.plot_comparison_prices(graph_comparisons_defaults,
#                            prices,
#                            suptitle="Graph comparisons - lambda_star="
#                                     + str(params["lambda_star"])
#                                     + "- 100 banks - 10 simus per graph")
#
#
#
#
# ### STATS DESC ON BALANCE SHEETS INITIALIZATION
# cycle_graph = GI.GraphInit(nx.cycle_graph(n))
# complete_graph = GI.GraphInit(nx.complete_graph(n))
# star_graph = GI.GraphInit(nx.star_graph(n-1))
# er_graph = GI.GraphInit(nx.erdos_renyi_graph(n, 0.05))
# graphs_dict = {}
# graphs_dict["ER 0.05"] = er_graph
# graphs_dict["Complete"] = complete_graph
# graphs_dict["Star"] = star_graph
# graphs_dict["Circle"] = cycle_graph
# # On average 1 edge out of 2 is negative and 1 out of 2 is positive
# p = 0.5
# # Values of loans and their respective probabilities
# vals = np.array([l])
# distrib = np.array([1])
# n_mc = 1000
#
# compinit = InitA.compare_initializations(params,
#                                   x0,
#                                   mus,
#                                   graphs_dict,
#                                   n_mc,
#                                   p,
#                                   vals,
#                                   distrib)
#
# compinit_equities = InitA.get_consodict_equities(compinit)
# compinit_loans = InitA.get_consodict_loans(compinit)
# compinit_debts = InitA.get_consodict_debts(compinit)
# compinit_assets = InitA.get_consodict_assets(compinit)
# compinit_reserves = InitA.get_consodict_reserves(compinit)
# compinit_portfolios = InitA.get_consodict_portfolios(compinit)
#
# Viz.plot_init_hists(compinit_portfolios, "portfolios values")
# Viz.plot_init_hists(compinit_equities, "equities")
# Viz.plot_init_hists(compinit_loans, "loans")
# Viz.plot_init_hists(compinit_debts, "debts")
# Viz.plot_init_hists(compinit_reserves, "reserves")
# Viz.plot_init_hists(compinit_assets, "assets")
