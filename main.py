import numpy as np
import networkx as nx
import BankNetwork
import RiskyAssets
import importlib
import matplotlib.pyplot as plt
import time
import BalanceSheetInit as BSI
import GraphInit as GI
import SimulationsTools as ST
importlib.reload(ST)
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(BSI)


#### Dictionary of parameters
params = dict()


#### Fundamental parameters
n = 100
m = 4
T = 1000
params["m"] = m
params["T"] = T
params["n"] = n
# params["r"] = 0.02
r_annual = 0.1
params["r"] = ST.daily_compound(r_annual, 365)
params["xi"] = 0.5
params["zeta"] = 0.5
params["lambda_star"] = 10


### RISKY ASSETS PARAMETERS
x0 = 10
mu = 0.005
sigma = 0.5
init_val = x0 * np.ones((m, ))
mus = mu * np.ones((m, ))
sigmas = sigma * np.ones((m, ))
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
qinit = BSI.QInit(n, ws)
params["q"] = qinit.random_asset_choice()


### BALANCE SHEET INITIALIZATION PARAMETERS
params["liquidator"] = True
# Nominal value of all loans (and debts)
l = 10000
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


### RANDOM GRAPH INITIALIZATION
# On average 1 edge out of 2 is negative and 1 out of 2 is positive
p = 0.5
# Values of loans and their respective probabilities
vals = np.array([l])
distrib = np.array([1])
# Graph structure
#graph = nx.cycle_graph(n)
graph = nx.erdos_renyi_graph(n, 0.01)
# graph = nx.complete_graph(n)
graph = GI.GraphInit(graph)
# Number of Monte Carlo iterations
n_mc = 10
# MC on random allocations on graph
mc_list = ST.mc_on_graphs(params, prices, x0, mus, graph, n_mc, p, vals, distrib)























############################JUNK####################################"
p = 0.5
# graph = nx.erdos_renyi_graph(n, 0.2)
graph = nx.complete_graph(n)
graph = GI.GraphInit(graph)
vals = np.array([l])
distrib = np.array([1])
L = random_allocation(graph, p, vals, distrib)
init_bs = BSI.BalanceSheetInit(L,
                               params["r"],
                               params["q"],
                               params["alphas"],
                               params["betas"],
                               params["lambda_star"],
                               x0,
                               mus)





equities = mc_list[0].get_equities_record()
portfolios = mc_list[0].get_portfolios_record()
loans = mc_list[0].get_loans_record()
debts = mc_list[0].get_debts_record()


mc_mean = np.zeros((T, ))
for s in range(0, n_mc):
    net = mc_list[s]
    defaulting = net.get_defaulting_record()
    cum_defaulting = np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])])
    mc_mean += (1 / n_mc) * np.cumsum(cum_defaulting)
mc_mean = mc_mean.astype(np.float64)

plt.plot(mc_mean)


mc_losses = np.zeros((T, ))
for s in range(0, n_mc):
    net = mc_list[s]
    losses = net.lost_value
    mc_losses += (1 / n_mc) * np.cumsum(losses)
mc_losses = mc_losses.astype(np.float64)


mc_er006
mc_er01
mc_er03


fig, axes = plt.subplots(2)
axes[0].plot(mc_er03_r0, label="r=0")
axes[0].plot(mc_er03_r1, label="r=0.01")
axes[0].plot(mc_mean, label="r=0.015")
axes[0].plot(mc_er03, label="r=0.02")


axes[1].plot(prices[:, 0])
axes[1].plot(prices[:, 1])

axes[0].set_ylabel("Cumulated defaults")

axes[1].set_ylabel("Price")

axes[1].set_xlabel("Time")

plt.suptitle("ER p=0.3 graph - interest rate's influence - 100 simulations - 100 banks")
axes[1].set_title("")




# mc_er03_r0 : r=0
# mc_er03 : r=0.02
# mc_er03_r1 : r=0.01

fig, axes = plt.subplots(2)
net = mc_list[4]
defaulting = net.get_defaulting_record()
cum_defaulting = np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])])
axes[0].plot(np.cumsum(cum_defaulting))


axes[1].plot(prices[:, 0])
axes[1].plot(prices[:, 1])

axes[0].set_ylabel("Cumulated defaults")

axes[1].set_ylabel("Price")

axes[1].set_xlabel("Time")

plt.suptitle("ER p=0.3 graph - A few initial loans allocations scenarios")
axes[1].set_title("")