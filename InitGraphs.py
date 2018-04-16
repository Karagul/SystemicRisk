import numpy as np
import networkx
import scipy
import  BankNetwork
import RiskyAssets
import importlib
import matplotlib.pyplot as plt
import time
importlib.reload(BankNetwork)



testgraph = networkx.powerlaw_cluster_graph(10000, 100, 0.1)
adjacency = networkx.to_numpy_matrix(testgraph)
# networkx.draw(testgraph)

cycle = networkx.cycle_graph(10)
networkx.draw(cycle)

star = networkx.star_graph(10)
networkx.draw(star)

complete = networkx.complete_graph(10)
networkx.draw(complete)
a = networkx.to_numpy_matrix(complete)


def vec_gene(v):
    """
    Generate submatrix from vector used in the construction of the expanded adjacency matrix
    Args:
        v (numpy.ndarray) : the generating vector
    Returns:
        numpy.ndarray : the generated matrix
    """
    mat = np.zeros((v.shape[0] + 1, v.shape[0]))
    mat[0, :] = v
    mat[1:, :] = np.diag(v)
    return mat


def expanded_adjacency(A):
    n = A.shape[0]
    n_exp = (n * (n - 1)) / 2
    A_exp = np.zeros((n, int(n_exp)))
    col = 0
    for i in range(1, n):
        v = A[i - 1, i:]
        print(v)
        mat = vec_gene(v)
        A_exp[i - 1:, col : col + n - i] = mat
        col += n - i
    return A_exp


def weights_allocation(A, Y):
    A_exp = expanded_adjacency(A)
    pinv = np.linalg.pinv(A_exp)
    sol = np.dot(pinv, Y)
    return sol

complete = networkx.complete_graph(5)
A_complete = networkx.to_numpy_matrix(complete)
A_complete = np.asarray(A_complete)

star = networkx.star_graph(4)
A_star = networkx.to_numpy_matrix(star)
A_star = np.asarray(A_star)

er = networkx.erdos_renyi_graph(5, 0.5)
networkx.draw(er)
A_er = networkx.to_numpy_matrix(er)
A_er = np.asarray(A_er)

cycle = networkx.cycle_graph(5)
A_cycle = networkx.to_numpy_matrix(cycle)
A_cycle = np.asarray(A_cycle)


y = np.zeros((5, 1))
y[0, 0] = 100
y[1, 0] = - 100
y[2, 0] = 200
y[3, 0] = -100
y[4, 0] = - 100


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
n = 1000
ld = 5000
eq = 10000
tau = 0.5
m = 4
p0 = 10
#L, R, Q = complete_init(n, ld, eq, tau, m, p0)
L, R, Q = cycle_init(n, ld, eq, tau, m, p0)
# D = networkx.DiGraph(L)
# networkx.draw(D, pos=networkx.circular_layout(D))
alpha = 0.25
alphas = 0.25 * np.ones((n, ))
r = 0.02
xi = 0.6
zeta = 0.6
bar_E = 5000 * np.ones((n, ))

init_val = np.array([10, 10, 10, 10])
mus = np.array([0.01, 0.01, 0.01, 0.01])
sigmas = np.array([0.5, 0.5, 0.5, 0.5])
assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
prices = assets.generate()

test = BankNetwork.BankNetwork(L, R, Q, alphas, r, xi, zeta, bar_E)
test.add_liquidator()
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


plt.plot(cum_defaulting)



rsvs = test.get_equities_record()
plt.plot(rsvs[0, :])
plt.plot(rsvs[1, :])
plt.plot(rsvs[2, :])
plt.plot(rsvs[3, :])
plt.plot(rsvs[4, :])
plt.plot(rsvs[5, :])
plt.plot(rsvs[6, :])
plt.plot(rsvs[7, :])



eqts = test.get_equities_record()
plt.plot(eqts[0, :])
plt.plot(eqts[1, :])
plt.plot(eqts[2, :])
plt.plot(eqts[3, :])
plt.plot(eqts[4, :])
plt.plot(eqts[5, :])