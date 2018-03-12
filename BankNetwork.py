import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class BankNetwork:

    def __init__(self, L, R, Q, alpha, r, xi, zeta, bar_E=None):
        self.L = L
        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.P = None
        self.E = None
        if not bar_E:
            self.bar_E = np.zeros((self.L.shape[0], ))
        else:
            self.bar_E = bar_E
        self.Psi = None
        self.Pi = None
        self.r = r
        self.xi = xi
        self.zeta = zeta
        self.liquidator = False
        self.record = []
        self.period = 0
        self.defaulted = np.zeros((self.L.shape[0], ))

    def add_liquidator(self):
        n = self.L.shape[0]
        z = np.zeros((n, 1))
        self.L = np.concatenate((z, self.L), axis=1)
        z = np.zeros((1, n + 1))
        self.L = np.concatenate((z, self.L), axis=0)
        zq = np.zeros((1, self.Q.shape[1]))
        self.Q = np.concatenate((zq, self.Q))
        zv = np.zeros((1, ))
        self.R = np.concatenate((zv, self.R))
        self.bar_E = np.concatenate((zv, self.bar_E))
        self.defaulted = np.concatenate((zv, self.defaulted))
        self.liquidator = True

    def net_loans_matrix(self):
        self.L = np.maximum(self.L - self.L.T, np.zeros(shape=self.L.shape))

    def get_loans(self):
        return np.sum(self.L, axis=0)

    def get_debts(self):
        return np.sum(self.L, axis=1).T

    def get_loans_matrix(self):
        return self.L

    def get_equities(self):
        return self.E

    def get_portfolios(self):
        return self.P

    def get_psi(self):
        return self.Psi

    def get_pi(self):
        return self.Pi

    def get_reserves(self):
        return self.R

    def get_defaulted(self):
        return self.defaulted

    def get_defaulting(self):
        defaulting = np.greater_equal(self.bar_E - self.E, 0).astype(np.int64)
        return defaulting - self.defaulted

    def update_defaulted(self):
        self.defaulted = np.maximum(self.defaulted, self.get_defaulting())

    def update_portfolios(self, X):
        P = np.dot(self.Q, X)
        self.P = P

    def update_reserves(self):
        self.R += self.r * (self.get_loans() - self.get_debts()) + \
                  np.dot(self.Psi, self.Pi) * self.get_defaulting()

    def update_equities(self):
        self.E = self.R + self.P + self.get_loans() - self.get_debts()

    def compute_psi(self):
        self.Psi = normalize(self.L, axis=0, norm="l1")

    def compute_pi(self):
        self.Pi = self.xi * self.P + self.R + self.zeta * self.get_loans()

    def zero_out(self, j):
        for k in range(0, self.L.shape[0]):
            self.L[j, k] = 0
            self.L[k, j] = 0
        self.E[j] = 0
        self.Q[j, :] = np.zeros((self.Q.shape[1], ))
        self.P[j] = 0
        self.R[j] = 0

    def update_liquidator(self):
        defaulting = self.get_defaulting()
        loans_defaulting = np.dot(defaulting, self.get_loans())
        defaulting_index = np.argwhere(defaulting == 1)
        self.R[0] -= self.zeta * loans_defaulting
        self.E[0] += (1 - self.zeta) * loans_defaulting
        for j in defaulting_index :
            for k in range(0, self.L.shape[0]):
                self.L[0, k] += self.L[j, k]

    def liquidate_liquidator(self):
        defaulting = self.get_defaulting()
        defaulting_index = np.argwhere(defaulting == 1)
        if defaulting.sum() >= 1 :
            self.compute_psi()
            self.compute_pi()
            print(self.Pi)
        self.update_liquidator()
        for j in defaulting_index:
            self.zero_out(j)

        


   # def liquidate_internal(self):

    def snap_record(self):
        rec_dic = dict()
        rec_dic["L"] = self.L.copy()
        rec_dic["R"] = self.R.copy()
        rec_dic["Q"] = self.Q.copy()
        rec_dic["P"] = self.P.copy()
        rec_dic["E"] = self.E.copy()
        self.record.append(rec_dic)
