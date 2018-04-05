import numpy as np
import importlib
import BankNetwork

importlib.reload(BankNetwork)

# Initialization of toy parameters
r = 0.03
xi = 0.7
zeta = 0.7
n = 4
m = 2
R0 = 1000
mL = 1000
stdL = 100
mQ = 500
stdQ = 250
alpha0 = 0.4
L = np.random.normal(mL, stdL, (n, n))
R = R0 * np.ones(shape=(n, ))
Q = np.absolute(np.random.normal(mQ, stdQ, (n, m)))
alphas = alpha0 * np.ones(shape=(n, ))

# Initialization of the BankNetwork
network = BankNetwork.BankNetwork(L, R, Q, alphas, r, xi, zeta)

# Test for the class methods
network.add_liquidator()
network.net_loans_matrix()
print(network.get_loans())
print(network.get_debts())


# Test for protfolio update
prices = np.random.normal(10, 1, m)
network.update_portfolios(prices)
print(network.Q)
print(network.P)


# Test for defaulting and defaulted
network.update_defaulted()
# Test for equity update
print(network.get_equities())
print(network.get_defaulting())
print(network.get_defaulted())

prices = np.random.normal(10, 1, m)
network.stage2()
print(network.Pi)
network.stage1(prices)
network.stage2()
# print(network.P)
print(network.get_assets())
network.stage3()
#print(network.P)
print(network.get_assets())





