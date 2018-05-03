import numpy as np
import BankNetwork
import importlib
import scipy.optimize as optimize
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

    def get_equities(self):
        return self.E

    def get_star_equities(self):
        return self.get_debts() / (self.lambda_star * self.betas - 1)

    def get_positivity_threshold(self):
        return self.get_loans() - self.get_debts()

    def get_tilde_equities(self):
        tilde_mus_inv = 1 / self.get_tilde_mus()
        temp = self.r * self.x0 * tilde_mus_inv + self.r - 1
        return (self.get_debts() - self.get_loans()) * temp

    def get_minimal_equities(self):
        return np.maximum(
            np.maximum(
                self.get_star_equities(),
                self.get_tilde_equities()),
            self.get_positivity_threshold())

    def set_minimal_equities(self):
        self.E = self.get_minimal_equities()

    def set_manual_equities(self, E):
        minimal_equities = self.get_minimal_equities()
        self.E = np.maximum(minimal_equities, E)

    def augment_equities(self, bonus):
        self.E += bonus

    def get_alpha_thresholds(self):
        return self.r * self.x0 * \
            (self.get_debts() - self.get_loans()) / (self.get_assets() * self.get_tilde_mus())

    def get_portfolios(self):
        vec1 = self.E + (self.get_debts() - self.get_loans()) - \
            self.r * np.maximum((self.get_debts() - self.get_loans()), 0)
        vec2 = self.alphas * self.get_assets()
        return np.minimum(vec1, vec2)

    def get_reserves(self):
        return self.E - self.get_loans() + self.get_debts() - self.get_portfolios()

    def get_quantitities(self):
        Ps = self.get_portfolios()
        return (1 / self.x0) * self.q * Ps.reshape((Ps.shape[0], 1))


class QInit:

    def __init__(self, n, ws):
        """
        Initialize randomly the assets choices of banks using given weights (enablde to controle concentration/diversification
        in terms of assets.

        Params:
            n (int) : n banks
            ws (numpy.ndarray) : the weights to randomly allocate to assets, sum(ws) = 1, m will be inferred from ws.shape[0]
        """
        self.m = ws.shape[0]
        self.n = n
        self.ws = ws
        self.qs = np.zeros((n, self.m))

    def get_normalized_entropy(self):
        """
        Compute normalized entropy for self.ws vector. Gives an idea of concentration of investments

        Returns :
            float : the normalized entropy for self.ws
        """
        non_zeros = self.ws[self.ws > 0]
        unif = (1 / self.n) * np.ones((self.n, ))
        max_entropy = - np.sum(np.log(unif) * unif)
        return (- np.sum(np.log(non_zeros) * non_zeros)) / max_entropy

    def random_asset_choice(self):
        """
        Generate random allocation of assets using the weights self.ws

        Returns :
            numpy.ndarray : matrix of assets investment weights for each bank (one column per bank)
        """
        for i in range(0, self.n):
            cop = self.ws.copy()
            np.random.shuffle(cop)
            self.qs[i, :] = cop
        return self.qs
