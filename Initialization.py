import numpy as np
import BankNetwork
import importlib
importlib.reload(BankNetwork)


class Initialization:

    def __init__(self, L, r, q, alphas, betas, lambda_star, x0, mus):
        self.L = L
        self.alphas = alphas
        self.betas = betas
        self.lambda_star = lambda_star
        self.q = q
        self.r = r
        self.x0 = x0
        self.mus = mus

    def get_tilde_mus(self):
        return np.dot(self.q, self.mus)

    def get_loans(self):
        return np.sum(self.L, axis=1).T

    def get_debts(self):
        return np.sum(self.L, axis=0)

    def get_star_equities(self):
        return self.get_debts() / (self.lambda_star * self.betas - 1)

    def get_tilde_equities(self):
        tilde_mus_inv = 1 / self.get_tilde_mus()
        temp = self.r * self.x0 * tilde_mus_inv + self.r - 1
        return (self.get_debts() - self.get_loans()) * temp