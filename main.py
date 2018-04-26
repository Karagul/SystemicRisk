import numpy as np
import networkx as nx
import scipy
import BankNetwork
import RiskyAssets
import importlib
import matplotlib.pyplot as plt
import time
import BalanceSheetInit as BSI
import GraphInit as GI
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(BSI)


def random_allocation(graph, p, vals, distrib):
    bers = graph.generate_dibernouilli(p)
    vals = graph.generate_loans_values(vals, distrib)
    graph.set_loans(bers * vals)
    # Initialization of the balance sheets
    L = graph.get_loans_matrix()
    return L


def balance_sheet_allocation(params_dict, L, x0, mus):
    init_bs = BSI.BalanceSheetInit(L,
                                   params_dict["r"],
                                   params_dict["q"],
                                   params_dict["alphas"],
                                   params_dict["betas"],
                                   params_dict["lambda_star"],
                                   x0,
                                   mus)
    E = params_dict["e"] * np.ones((n,))
    init_bs.set_manual_equities(E)
    R = init_bs.get_reserves()
    Q = init_bs.get_quantitities()
    return R, Q


def mc_on_graphs(params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib):
    mc_list = []
    for s in range(0, n_mc):
        L = random_allocation(graph, p, vals, distrib)
        R, Q = balance_sheet_allocation(params_dict, L, x0, mus)
        # Creation of the bank network
        bank_network = BankNetwork.BankNetwork(L,
                                               R,
                                               Q,
                                               params_dict["alphas"],
                                               params_dict["r"],
                                               params_dict["xi"],
                                               params_dict["zeta"],
                                               params_dict["bar_E"])
        if params_dict["liquidator"]:
            bank_network.add_liquidator()
        bank_network.update_portfolios(prices[0, :])
        bank_network.compute_psi()
        bank_network.compute_pi()
        for t in range(0, T):
            bank_network.stage1(prices[t, :])
            bank_network.stage2()
            bank_network.stage3()
            bank_network.snap_record()
        mc_list.append(bank_network)
    return mc_list



#### Dictionary of parameters
params = dict()


#### Fundamental parameters
n = 100
m = 2
T = 1000
params["m"] = m
params["T"] = T
params["n"] = n
params["r"] = 0.02
params["xi"] = 0.7
params["zeta"] = 0.7
params["lambda_star"] = 2


#### Risky assets parameters
x0 = 10
mu = 0.005
sigma = 0.5
init_val = x0 * np.ones((m, ))
mus = mu * np.ones((m, ))
sigmas = sigma * np.ones((m, ))
assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
prices = assets.generate()
plt.figure()
plt.plot(prices[:, 0])
plt.plot(prices[:, 1])


#### Choices of risky assets :
# Max diversification
# ws = (1 / m) * np.ones((m, ))
# Min diversification
ws = np.zeros((m, ))
ws[0] = 1
qinit = BSI.QInit(n, ws)
params["q"] = qinit.random_asset_choice()


#### Balance sheets initialization parameters
params["liquidator"] = True
# Nominal value of all loans (and debts)
params["l"] = 5000
# Minimal equity for all banks
params["e"] = 10000
# Value of alpha parameter for all banks
alpha = 0.25
# Value of beta parameter for all banks
beta = 0.75
# Value of \bar E parameter for all banks
bar_e = 1000
params["alphas"] = alpha * np.ones((n, ))
params["betas"] = beta * np.ones((n, ))
params["bar_E"] = bar_e * np.ones((n, ))


# On average 1 edge out of 2 is negative and 1 out of 2 is positive
p = 0.5
# Values of loans and their respective probabilities
vals = np.array([5000])
distrib = np.array([1])
# Graph structure
graph = nx.cycle_graph(n)
graph = GI.GraphInit(graph)
# Number of Monte Carlo iterations
n_mc = 100

# MC on random allocations on graph
mc_list = mc_on_graphs(params, prices, x0, mus, graph, n_mc, p, vals, distrib)
