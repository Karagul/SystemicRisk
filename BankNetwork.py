import numpy as np


class BankNetwork:

    def __init__(self, L, R, Q, alpha, record=False):
        self.L = L
        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.record = record
        self.P = None
        self.E = None
        if record:
            self.Phist = []
            self.Rhist = []
            self.Ehist = []

    def add_liquidator(self):
        n = self.L.shape[0]
        z = np.zeros((n, 1))
        self.L = np.concatenate((z, self.L), axis=0)
        z = np.zeros((1, n+1))
        self.L = np.concatenate((z, self.L), axis=1)

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
        if self.record:
            self.Phist.append(self.P)
            self.Ehist.append(self.E)

    def compute_equities(self):
        self.E = self.R + self.P + self.get_loans() - self.get_debts()