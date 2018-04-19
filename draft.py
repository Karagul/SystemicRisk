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


T = 5000
n = 10
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
graph = networkx.complete_graph(n)
graph_init = GI.GraphInit(graph)
nedges = graph_init.get_nedges()
bers = np.random.binomial(2, p, nedges)
norms = np.random.normal(mean_l, std_l, nedges)
graph_init.set_loans(bers * norms)




init_val = x0 * np.ones((m, ))
mus = np.array([0.005, 0.005, 0.005, 0.005])
sigmas = np.array([0.5, 0.5, 0.5, 0.5])
assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
prices = assets.generate()


importlib.reload(BalanceSheetInit)
test = BalanceSheetInit.BalanceSheetInit(L, r, q, alphas, betas, lambda_star, p0, mus)
test.get_tilde_mus()
test.get_star_equities()
test.get_tilde_equities()
test.set_minimal_equities()
print(test.get_portfolios())
print(test.get_reserves())
test.get_quantitities()