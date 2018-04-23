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



def random_asset_choice(n, m):
    rand = np.random.randint(0, m, n)
    Q = np.zeros((n, m))
    for i in range(0, n):
        Q[i, rand[i]] = 1
    return Q


T = 1000
n = 1000
ld = 5000
eq = 10000
tau = 0.5
m = 4
x0 = 10
Q = random_asset_choice(n, m)
q = (1 / np.max(Q)) * Q
alpha = 0.25
alphas = alpha * np.ones((n, ))
beta = 0.75
betas = beta * np.ones((n, ))
r = 0.02
xi = 0.6
zeta = 0.6
bar_E = 5000 * np.ones((n, ))
lambda_star = 2


p = 0.5
mean_l, std_l = 5000, 100
# graph = networkx.complete_graph(n)
graph = networkx.cycle_graph(n)
graph_init = GI.GraphInit(graph)
nedges = graph_init.get_nedges()
bers = 2 * (np.random.binomial(1, p, nedges) - 0.5)
norms = np.random.normal(mean_l, std_l, nedges)
graph_init.set_loans(bers * norms)
L = graph_init.get_loans_matrix()


init_val = x0 * np.ones((m, ))
mus = np.array([0.005, 0.005, 0.005, 0.005])
sigmas = np.array([0.5, 0.5, 0.5, 0.5])
assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
prices = assets.generate()


init_bs = BSI.BalanceSheetInit(L, r, q, alphas, betas, lambda_star, x0, mus)
# test.get_tilde_mus()
# test.get_star_equities()
# test.get_tilde_equities()
init_bs.set_minimal_equities()
R = init_bs.get_reserves()
Q = init_bs.get_quantitities()

test = BankNetwork.BankNetwork(L, R, Q, alphas, r, xi, zeta, bar_E)

test.add_liquidator()

start  = time.clock()
test.update_portfolios(prices[0, :])
test.compute_psi()
test.compute_pi()


for t in range(0, T):
    test.stage1(prices[t, :])
    test.stage2()
    test.stage3()
    test.snap_record()

end = time.clock()
print(end - start)

defaulting = test.get_defaulting_record()
cum_defaulting = [np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])]

plt.figure()
plt.plot(np.cumsum(cum_defaulting))


rsvs = test.get_equities_record()
plt.plot(rsvs[0, :])
plt.plot(rsvs[1, :])
plt.plot(rsvs[2, :])
plt.plot(rsvs[3, :])
plt.plot(rsvs[4, :])
plt.plot(rsvs[5, :])
plt.plot(rsvs[6, :])
plt.plot(rsvs[7, :])
