import numpy as np
import networkx
import scipy
import BankNetwork
import RiskyAssets
import importlib
import matplotlib.pyplot as plt
import time
import BalanceSheetInit as BSI
import GraphInit as GI
from sklearn.preprocessing import normalize
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(BSI)



def random_asset_choice(n, m):
    rand = np.random.randint(0, m, n)
    Q = np.zeros((n, m))
    for i in range(0, n):
        Q[i, rand[i]] = 1
    return Q

# Fundamental parameters
T = 1000
n = 100
m = 2
r = 0.02
xi = 0.7
zeta = 0.7
lambda_star = 2


# Draws of risky assets
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


# Balance sheets initialization parameters
liquidator = True
ld = 5000
eq = 10000
alpha = 0.25
beta = 0.75
bar_eq = 1000
Q = random_asset_choice(n, m)
q = (1 / np.max(Q)) * Q
alphas = alpha * np.ones((n, ))
betas = beta * np.ones((n, ))
bar_E = bar_eq * np.ones((n, ))



# MC for a given graph structure
n_mc = 100
mc_list = []

# graph = networkx.complete_graph(n)
# graph = networkx.star_graph(n-1)
# graph = networkx.erdos_renyi_graph(n, 0.3)
graph = networkx.cycle_graph(n)

for s in range(0, n_mc) :

    # Choice of the graph structure
    graph_init = GI.GraphInit(graph)
    nedges = graph_init.get_nedges()
    # Random determination of the direction and weights
    p = 0.5
    bers = 2 * (np.random.binomial(1, p, nedges) - 0.5)
    loans_value = ld * np.ones((nedges, ))
    graph_init.set_loans(bers * loans_value)
    L = graph_init.get_loans_matrix()

    # Initialization of the balance sheets
    init_bs = BSI.BalanceSheetInit(L, r, q, alphas, betas, lambda_star, x0, mus)
    E = eq * np.ones((n, ))
    init_bs.set_manual_equities(E)
    R = init_bs.get_reserves()
    Q = init_bs.get_quantitities()

    # Creation of the bank network
    bank_network = BankNetwork.BankNetwork(L, R, Q, alphas, r, xi, zeta, bar_E)

    if liquidator :
        bank_network.add_liquidator()

    start  = time.clock()
    bank_network.update_portfolios(prices[0, :])
    bank_network.compute_psi()
    bank_network.compute_pi()

    for t in range(0, T):
        bank_network.stage1(prices[t, :])
        bank_network.stage2()
        bank_network.stage3()
        bank_network.snap_record()

    mc_list.append(bank_network)
    end = time.clock()
    print(end - start)

mc_mean = np.zeros((T, ))
for s in range(0, n_mc):
    net = mc_list[s]
    defaulting = net.get_defaulting_record()
    cum_defaulting = np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])])
    mc_mean += (1 / n_mc) * np.cumsum(cum_defaulting)
mc_mean = mc_mean.astype(np.float64)

fig, axes = plt.subplots(2)
axes[0].plot(mc_circle, label="Circle Graph")
axes[0].plot(mc_complete, label="Complete Graph")
axes[0].plot(mc_er, label="Erdos-Reyni p=0.3")
axes[0].plot(mc_star, label="Star Graph")
axes[0].legend()
axes[0].set_ylabel("Mean of cumulative defaults")
axes[0].set_xlabel("")
plt.suptitle("100 random draws of edges direction per graph type - 100 banks")

axes[1].plot(prices[:, 0])
axes[1].plot(prices[:, 1])
axes[1].set_title("Assets' prices")
axes[1].set_ylabel("Price")
axes[1].set_xlabel("Time")
