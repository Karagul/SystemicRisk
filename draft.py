import numpy as np
import networkx
import scipy
import BankNetwork
import RiskyAssets
import importlib
import matplotlib.pyplot as plt
import time
import Initialization
from sklearn.preprocessing import normalize
importlib.reload(BankNetwork)


def complete_allocation(n, c):
    L = np.zeros((n, n))
    for k in range(0, n):
        for s in range(0, int((n-1)/2)):
            L[k, (k + 1 + s) % n] = 1
    return c * L


def cycle_allocation(n, c):
    L = np.zeros((n, n))
    for k in range(0, n):
        L[k, (k + 1) % n] = 1
    return c * L


def random_asset_choice(n, m):
    rand = np.random.randint(0, m, n)
    Q = np.zeros((n, m))
    for i in range(0, n):
        Q[i, rand[i]] = 1
    return Q


def complete_init(n, ld, eq, tau, m, p0):
    E = eq * np.ones((n, ))
    L = complete_allocation(n, ld)
    R = tau * E
    q = (1 - tau) * eq / p0
    Q = q * random_asset_choice(n, m)
    return L, R, Q


def cycle_init(n, ld, eq, tau, m, p0):
    E = eq * np.ones((n, ))
    L = cycle_allocation(n, ld)
    R = tau * E
    q = (1 - tau) * eq / p0
    Q = q * random_asset_choice(n, m)
    return L, R, Q

importlib.reload(BankNetwork)

start = time.clock()

T = 5000
n = 10
ld = 5000
eq = 10000
tau = 0.5
m = 4
p0 = 10
#L, R, Q = complete_init(n, ld, eq, tau, m, p0)
L, R, Q = cycle_init(n, ld, eq, tau, m, p0)
q = (1 / np.max(Q)) * Q
# D = networkx.DiGraph(L)
# networkx.draw(D, pos=networkx.circular_layout(D))
alpha = 0.25
alphas = alpha * np.ones((n, ))
beta = 0.75
betas = beta * np.ones((n, ))

r = 0.02
xi = 0.6
zeta = 0.6
bar_E = 5000 * np.ones((n, ))
lambda_star = 2
mus = np.array([0.04, 0.01, 0.01, 0.03])

importlib.reload(Initialization)
test = Initialization.Initialization(L, r, q, alphas, betas, lambda_star, p0, mus)
test.get_tilde_mus()
test.get_star_equities()
test.get_tilde_equities()