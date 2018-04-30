import numpy as np
import time
from sklearn.preprocessing import normalize


class BankNetwork:

    def __init__(self, L, R, Q, alpha, r, xi, zeta, bar_E=None, lambda_star=None):
        self.L = L
        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.lambda_star = lambda_star
        self.P = None
        if not isinstance(bar_E, np.ndarray):
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
        self.defaulted = np.zeros((self.L.shape[0], ))
        self.defaulting = np.zeros((self.L.shape[0],))
        self.lost_value = []
        self.track = []

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
        self.defaulting = np.concatenate((zv, self.defaulting))
        self.liquidator = True

    def net_loans_matrix(self):
        self.L = np.maximum(self.L - self.L.T, np.zeros(shape=self.L.shape))

    def get_assets(self):
        return self.R + self.P + self.get_loans()

    def get_m(self):
        return self.Q.shape[1]

    def get_n(self):
        return self.R.shape[0] - int(self.liquidator)

    def get_loans(self):
        return np.sum(self.L, axis=1).T

    def get_debts(self):
        return np.sum(self.L, axis=0)

    def get_loans_matrix(self):
        return self.L

    def get_equities(self):
        eq = np.maximum(self.get_assets() - self.get_debts(), 0)
        eq[np.absolute(eq) < 1e-10] = 0
        return eq

    def get_portfolios(self):
        return self.P

    def get_psi(self):
        return self.Psi

    def get_pi(self):
        return self.Pi

    def get_reserves(self):
        return self.R

    def get_leverage(self):
        return self.get_assets() / self.get_equities()

    def get_defaulted(self):
        return self.defaulted

    def get_defaulting(self):
        return self.defaulting

    def update_defaulted(self):
        self.defaulted = np.maximum(self.defaulted, self.get_defaulting())

    def update_defaulting(self):
        self.defaulting = np.greater_equal(self.bar_E - self.get_equities(), 0).astype(np.int64)
        if self.lambda_star:
            leverages = self.get_leverage()
            self.defaulting += np.greater(leverages, self.lambda_star).astype(np.int64)
            self.defaulting[self.defaulting > 1] = 1
        if self.liquidator:
            self.defaulting[0] = 0
        self.defaulting = np.maximum(self.defaulting - self.defaulted, 0)

    def update_portfolios(self, X):
        P = np.dot(self.Q, X)
        self.P = P

    def update_reserves(self):
        self.R += self.r * (self.get_loans() - self.get_debts()) + \
                  np.dot(self.Psi, self.Pi)

    def get_non_defaulting(self):
        defaulting = self.get_defaulting()
        healthy = np.ones(defaulting.shape) - defaulting
        return healthy

    def non_default_loans(self):
        healthy = self.get_non_defaulting()
        return np.dot(self.L, healthy)

    def compute_psi(self):
        self.Psi = normalize(self.L, axis=0, norm="l1")

    def compute_pi(self):
        common = self.xi * self.P + self.R
        self.Pi = common + int(self.liquidator) * self.zeta * self.non_default_loans()
        self.Pi *= self.get_defaulting()

    def zero_out(self, j):
        for k in range(0, self.L.shape[0]):
            self.L[j, k] = 0
            self.lost_value[-1] += self.L[k, j]
            self.L[k, j] = 0
        self.Q[j, :] = np.zeros((self.Q.shape[1], ))
        self.lost_value[-1] += (1 - self.xi) * self.P[j]
        self.P[j] = 0
        self.R[j] = 0

    def update_liquidator(self):
        start = time.clock()
        defaulting = self.get_defaulting()
        if defaulting.sum() >= 1:
            loans_defaulting = np.dot(defaulting, self.non_default_loans())
            defaulting_index = np.argwhere(defaulting == 1)
            non_defaulting_index = np.argwhere(defaulting == 0)
            self.R[0] -= self.zeta * loans_defaulting
            for j in defaulting_index :
                for k in non_defaulting_index:
                    self.L[0, k] += self.L[j, k]
        end = time.clock()
        self.track.append(end - start)

    def loans_rewiring(self):
        start = time.clock()
        defaulting = self.get_defaulting()
        self.L += np.dot(self.Psi * defaulting, self.L)
        end = time.clock()
        self.track.append(end - start)

    def zero_out_defaulting(self):
        defaulting = self.get_defaulting()
        defaulting_index = np.argwhere(defaulting == 1)
        for j in defaulting_index:
            self.zero_out(j)

    def all_updates(self, X):
        self.update_reserves()
        self.update_portfolios(X)

    def managed_portfolio(self):
        liq_ind = int(self.liquidator)
        pnew = np.minimum(self.P[liq_ind:] + self.R[liq_ind:] - self.r * (self.get_debts()[liq_ind:] - self.get_loans()[liq_ind:]),
                          self.alpha * self.get_assets()[liq_ind:])
        non_defaulting = self.get_non_defaulting()
        return pnew * non_defaulting[liq_ind:]

    def stage1(self, X):
        self.lost_value.append(0)
        if self.liquidator:
            self.update_liquidator()
        else:
            self.loans_rewiring()
        self.zero_out_defaulting()
        self.all_updates(X)
        self.update_defaulted()
        self.update_defaulting()

    def stage2(self):
        self.compute_pi()
        self.compute_psi()

    def stage3(self):
        liq_ind = int(self.liquidator)
        new_p = self.managed_portfolio()
        management_vec = 1 + (1 / self.P[liq_ind:]) * (new_p - self.P[liq_ind:])
        np.place(management_vec, np.isnan(management_vec), 0)
        np.place(management_vec, np.isinf(management_vec), 0)
        management_matrix = np.repeat(management_vec.reshape((self.get_n(), 1)), self.get_m(), axis=1)
        self.Q[liq_ind:, :] *= management_matrix
        self.R[liq_ind:] += (self.P[liq_ind:] - new_p)
        self.P[liq_ind:] = new_p

    def snap_record(self):
        rec_dic = dict()
        rec_dic["L"] = self.L.copy()
        rec_dic["L+"] = self.get_loans().copy()
        rec_dic["D+"] = self.get_debts().copy()
        rec_dic["R"] = self.R.copy()
        rec_dic["Q"] = self.Q.copy()
        rec_dic["P"] = self.P.copy()
        rec_dic["E"] = self.get_equities()
        rec_dic["Defaulting"] = self.get_defaulting()
        rec_dic["Defaulted"] = self.defaulted.copy()
        self.record.append(rec_dic)

    def get_equities_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        equities_tuple = tuple([self.record[t]["E"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(equities_tuple, axis=1)

    def get_reserves_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        reserves_tuple = tuple([self.record[t]["R"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(reserves_tuple, axis=1)

    def get_portfolios_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        portfolios_tuple = tuple([self.record[t]["P"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(portfolios_tuple, axis=1)

    def get_loans_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        loans_tuple = tuple([self.record[t]["L+"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(loans_tuple, axis=1)

    def get_debts_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        debts_tuple = tuple([self.record[t]["D+"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(debts_tuple, axis=1)

    def get_defaulting_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        defaulting_tuple = tuple([self.record[t]["Defaulting"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(defaulting_tuple, axis=1)

    def get_defaults_cdf(self):
        defaulting = self.get_defaulting_record()
        cum_defaulting = np.cumsum(np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])]))
        return cum_defaulting