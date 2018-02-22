import numpy as np
import BankNetwork


r = 0.03
xi = 0.7
zeta = 0.7
n = 5
m = 2
R0 = 1000
mL = 1000
stdL = 100
mQ = 500
stdQ = 250
alpha0 = 0.4

L = np.random.normal(mL, stdL, (n, n))
R = R0 * np.ones(shape=(n, ))
Q = np.random.normal(mQ, stdQ, (n, m))
alphas = alpha0 * np.ones(shape=(n, ))

network = BankNetwork.BankNetwork(L, R, Q, alphas, r, xi, zeta)