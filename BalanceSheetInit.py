import numpy as np
import BankNetwork
import importlib
importlib.reload(BankNetwork)


class BalanceSheetInit:

    def __init__(self, L, r, q, alphas, betas, lambda_star, x0, mus):
        self.L = L
        self.alphas = alphas
        self.betas = betas
        self.lambda_star = lambda_star
        self.q = q
        self.r = r
        self.E = None
        self.x0 = x0
        self.mus = mus

    def get_tilde_mus(self):
        return np.dot(self.q, self.mus)

    def get_loans(self):
        return np.sum(self.L, axis=1).T

    def get_debts(self):
        return np.sum(self.L, axis=0)

    def get_assets(self):
        return self.E + self.get_debts()

    def get_star_equities(self):
        return self.get_debts() / (self.lambda_star * self.betas - 1)

    def get_tilde_equities(self):
        tilde_mus_inv = 1 / self.get_tilde_mus()
        temp = self.r * self.x0 * tilde_mus_inv + self.r - 1
        return (self.get_debts() - self.get_loans()) * temp

    def set_minimal_equities(self):
        self.E = np.maximum(self.get_star_equities(), self.get_tilde_equities())

    def set_manual_equities(self, E):
        minimal_equities = np.maximum(self.get_star_equities(), self.get_tilde_equities())
        self.E = np.maximum(minimal_equities, E)

    def get_alpha_thresholds(self):
        return self.r * self.x0 * (self.get_debts() - self.get_loans()) / (self.get_assets() * self.get_tilde_mus())

    def get_portfolios(self):
        vec1 = self.E + (1 - self.r) * (self.get_debts() - self.get_loans())
        vec2 = self.alphas * self.get_assets()
        return np.minimum(vec1, vec2)

    def get_reserves(self):
        return self.E - self.get_loans() + self.get_debts() - self.get_portfolios()

    def get_quantitities(self):
        Ps = self.get_portfolios()
        return (1 / self.x0) * self.q * Ps.reshape((Ps.shape[0], 1))
