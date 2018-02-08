import numpy as np


class BankNetwork:

    def __init__(self, L, R, Q, alpha):
        self.L = L
        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.P = None
        self.E = None
        self.liquidator = False
        self.record = []
        self.period = 0

    def add_liquidator(self):
        n = self.L.shape[0]
        z = np.zeros((n, 1))
        self.L = np.concatenate((z, self.L), axis=0)
        z = np.zeros((1, n+1))
        self.L = np.concatenate((z, self.L), axis=1)
        self.liquidator = True

    def get_loans(self):
        return np.sum(self.L, axis=0)

    def get_debts(self):
        return np.sum(self.L, axis=1).T

    def prices_update(self, X):
        P = np.dot(self.Q, X)
        if self.E :
            self.E -= self.P
            self.E += P
        else:
            self.E = self.R + P + self.get_loans() - self.get_debts()
        self.P = P

    def compute_equities(self):
        self.E = self.R + self.P + self.get_loans() - self.get_debts()

    def snap_record(self):
        rec_dic = dict()
        rec_dic["L"] = self.L.copy()
        rec_dic["R"] = self.R.copy()
        rec_dic["Q"] = self.Q.copy()
        rec_dic["P"] = self.P.copy()
        rec_dic["E"] = self.E.copy()
        self.record.append(rec_dic)

    def rebalance(self):

    def default_check(self):

