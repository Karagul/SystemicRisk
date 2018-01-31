import numpy as np


class BankNetwork:

    def __init__(self, L, R, Q, alpha):
        self.L = L
        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.P = None
        self.E = None

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

    def update_equities(self):
        self.E = self.R + self.P + self.get_loans() - self.get_debts()