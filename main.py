# Third party modules
import numpy as np
import networkx as nx
import BankNetwork
import RiskyAssets
import importlib
import time
import matplotlib.pyplot as plt

# Local imports
import BalanceSheetInit as BSI
import GraphInit as GI
import InitAnalysis as InitA
import SimulationsTools as ST
import Measures
import Visualization as Viz


importlib.reload(ST)
importlib.reload(Measures)
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(InitA)
importlib.reload(Viz)
importlib.reload(BSI)


#### Dictionary of parameters
params = dict()


#### Fundamental parameters
n = 100
m = 2
T = 3000
params["m"] = m
params["T"] = T
params["n"] = n
# params["r"] = 0.02
r_annual = 0.05
params["r"] = ST.daily_compound(r_annual, 365)
params["xi"] = 0.5
params["zeta"] = 0.5
params["lambda_star"] = 10


### RISKY ASSETS PARAMETERS
x0 = 10
mu = 0.01
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
# ws[0] = 1
qinit = BSI.QInit(n, ws)
params["q"] = qinit.random_asset_choice()


### BALANCE SHEET INITIALIZATION PARAMETERS
params["liquidator"] = True
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
# Number of Monte Carlo iterations for
n_mc_graph = 10
# MC on random allocations on graph
start = time.clock()
# mc_list = ST.mc_on_graphs(params, prices, x0, mus, graph, n_mc, p, vals, distrib)
end = time.clock()
print(end - start)





### MC on prices
# On average 1 edge out of 2 is negative and 1 out of 2 is positive
p = 0.5
p_er = 0.1

er_ps = [0.01, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

for p_er in er_ps :

# Values of loans and their respective probabilities
    vals = np.array([2*l/p_er])
    distrib = np.array([1])
# Graph structure
#graph = nx.cycle_graph(n)
    graph = nx.erdos_renyi_graph(n, p_er)
# graph = nx.complete_graph(n)
    graph = GI.GraphInit(graph)
# Number of Monte Carlo iterations for
    n_mc_graph = 1
    n_mc_prices = 1000
    prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)
    path = "E:/Simulations/ER" + str(p_er) + "_Leverage" + str(params["lambda_star"]) + "/"
    start = time.clock()
#    mc_dict = ST.mc_on_prices_ergraphs(params, prices_list, x0, mus, er_ps, n_mc_graph, p, vals, distrib)
    mc_prices = ST.mc_on_prices(params, prices_list, x0, mus, graph, n_mc_graph, p, vals, distrib, save_out=path)
    end = time.clock()
    print(end - start)







p_er=0.5
vals = np.array([2*l/p_er])
distrib = np.array([1])
# Graph structure
#graph = nx.cycle_graph(n)
graph = nx.erdos_renyi_graph(n, p_er)
# graph = nx.complete_graph(n)
graph = GI.GraphInit(graph)
# Number of Monte Carlo iterations for
n_mc_graph = 1
n_mc_prices = 1000
prices_list = ST.generate_prices(x0, m, mu, sigma, T, n_mc_prices)
path = "E:/Simulations/ER" + str(p_er) + "_Leverage" + str(params["lambda_star"]) + "/"
start = time.clock()
#    mc_dict = ST.mc_on_prices_ergraphs(params, prices_list, x0, mus, er_ps, n_mc_graph, p, vals, distrib)
mc_prices = ST.mc_on_prices(params, prices_list, x0, mus, graph, n_mc_graph, p, vals, distrib, save_out=path)
end = time.clock()
print(end - start)












# Comparison of several Erdos Reyni graphs
er_params = [0.01, 0.05, 0.2, 0.5, 0.8, 1]
er_comparisons = ST.compare_ER_graphs(params, prices, x0, mus, er_params, n_mc, p, vals, distrib)
er_defaults_cdf = Measures.average_defaults_cdf_dict(er_comparisons)
Viz.plot_comparison_prices(er_defaults_cdf,
                           prices,
                           suptitle="ER graphs, differents ps - Minimum asset diversification - lambda_star="
                                    + str(params["lambda_star"])
                                    + "- 100 banks - 10 simus per graph")


# Comparison of several graph structures
graphs_dict = dict()
cycle_graph = nx.cycle_graph(n)
complete_graph = nx.complete_graph(n)
star_graph = nx.star_graph(n-1)
er_graph = nx.erdos_renyi_graph(n, 0.05)
graphs_dict["Circle"] = GI.GraphInit(cycle_graph)
graphs_dict["Star"] = GI.GraphInit(star_graph)
graphs_dict["ER - 0.05"] = GI.GraphInit(er_graph)
graphs_dict["Complete"] = GI.GraphInit(complete_graph)
graph_comparisons = ST.compare_graphs(params, prices, x0, mus, graphs_dict, n_mc, p, vals, distrib)
graph_comparisons_defaults = Measures.average_defaults_cdf_dict(graph_comparisons)
Viz.plot_comparison_prices(graph_comparisons_defaults,
                           prices,
                           suptitle="Graph comparisons - lambda_star="
                                    + str(params["lambda_star"])
                                    + "- 100 banks - 10 simus per graph")




### STATS DESC ON BALANCE SHEETS INITIALIZATION
cycle_graph = GI.GraphInit(nx.cycle_graph(n))
complete_graph = GI.GraphInit(nx.complete_graph(n))
star_graph = GI.GraphInit(nx.star_graph(n-1))
er_graph = GI.GraphInit(nx.erdos_renyi_graph(n, 0.05))
graphs_dict = {}
graphs_dict["ER 0.05"] = er_graph
graphs_dict["Complete"] = complete_graph
graphs_dict["Star"] = star_graph
graphs_dict["Circle"] = cycle_graph
# On average 1 edge out of 2 is negative and 1 out of 2 is positive
p = 0.5
# Values of loans and their respective probabilities
vals = np.array([l])
distrib = np.array([1])
n_mc = 1000

compinit = InitA.compare_initializations(params,
                                  x0,
                                  mus,
                                  graphs_dict,
                                  n_mc,
                                  p,
                                  vals,
                                  distrib)

compinit_equities = InitA.get_consodict_equities(compinit)
compinit_loans = InitA.get_consodict_loans(compinit)
compinit_debts = InitA.get_consodict_debts(compinit)
compinit_assets = InitA.get_consodict_assets(compinit)
compinit_reserves = InitA.get_consodict_reserves(compinit)
compinit_portfolios = InitA.get_consodict_portfolios(compinit)

Viz.plot_init_hists(compinit_portfolios, "portfolios values")
Viz.plot_init_hists(compinit_equities, "equities")
Viz.plot_init_hists(compinit_loans, "loans")
Viz.plot_init_hists(compinit_debts, "debts")
Viz.plot_init_hists(compinit_reserves, "reserves")
Viz.plot_init_hists(compinit_assets, "assets")
