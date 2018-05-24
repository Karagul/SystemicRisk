""""*******************************************************
 * Copyright (C) 2017-2018 Dimitri Bouche dimi.bouche@gmail.com
 *
 * This file is part of an ongoing research project on systemic risk @CMLA (ENS Cachan)
 *
 * This file can not be copied and/or distributed without the express
 * permission of Dimitri Bouche.
 *******************************************************"""


import numpy as np
import time
from sklearn.preprocessing import normalize


class BankNetwork:

    def __init__(
            self,
            L,
            R,
            Q,
            alpha,
            r,
            xi,
            zeta,
            bar_E=None,
            lambda_star=None,
            enforce_leverage=True):
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
        self.enforce_leverage = enforce_leverage
        self.record = []
        self.defaulted = np.zeros((self.L.shape[0], ))
        self.defaulting = np.zeros((self.L.shape[0],))
        self.lost_value = []
        self.cumdefaults_leverage = []
        self.cumdefaults_classic = []
        self.leverage_counter = 0
        self.classic_counter = 0
        self.initial_value = None
        self.in_degrees = []
        self.out_degrees = []
        self.t = 0
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

    def set_initial_value(self):
        self.initial_value = np.sum(self.get_assets())

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
        lev = self.get_assets() / self.get_equities()
        return lev

    def get_defaulted(self):
        return self.defaulted

    def get_defaulting(self):
        return self.defaulting

    def get_lost_value(self):
        return self.lost_value

    def get_nodes_outdegree(self):
        return (self.L > 0).astype(int).sum(axis=1)

    def get_nodes_indegree(self):
        return (self.L > 0).astype(int).sum(axis=0)

    def update_defaulted(self):
        self.defaulted = np.maximum(self.defaulted, self.get_defaulting())

    def update_defaulting(self):
        classic = np.maximum(
            np.greater_equal(self.bar_E - self.get_equities(),
                             0).astype(np.int64) - self.defaulted,
            0)
        if self.enforce_leverage:
            leverages = self.get_leverage()
            leverage_linked = np.maximum(
                np.greater(leverages, self.lambda_star).astype(np.int64) - self.defaulted,
                0)
            leverage_linked = np.maximum(leverage_linked - classic, 0)
        else:
            leverage_linked = np.zeros((self.get_n() + int(self.liquidator)))
        if self.liquidator:
            leverage_linked[0] = 0
            classic[0] = 0
        self.leverage_counter += np.sum(leverage_linked)
        self.classic_counter += np.sum(classic)
        self.defaulting = leverage_linked + classic

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
        self.Pi = common + int(self.liquidator) * \
            self.zeta * self.non_default_loans()
        self.Pi *= self.get_defaulting()

    def zero_out(self, j):
        for k in range(0, self.L.shape[0]):
            self.L[j, k] = 0
            self.lost_value[-1] += self.L[k, j]
            if self.liquidator:
                self.lost_value[-1] += (1 - self.zeta) * self.L[k, j]
            self.L[k, j] = 0
        self.Q[j, :] = np.zeros((self.Q.shape[1], ))
        self.lost_value[-1] += (1 - self.xi) * self.P[j]
        self.lost_value[-1] -= self.R[j]
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
            for j in defaulting_index:
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
        pnew = np.minimum(self.P[liq_ind:] +
                          self.R[liq_ind:] -
                          self.r *
                          np.maximum((self.get_debts()[liq_ind:] -
                                      self.get_loans()[liq_ind:]), 0), self.alpha *
                          self.get_assets()[liq_ind:])
        non_defaulting = self.get_non_defaulting()
        return pnew * non_defaulting[liq_ind:]

    def stage1(self, X):
        self.t += 1
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
        management_vec = 1 + \
            (1 / self.P[liq_ind:]) * (new_p - self.P[liq_ind:])
        np.place(management_vec, np.isnan(management_vec), 0)
        np.place(management_vec, np.isinf(management_vec), 0)
        management_matrix = np.repeat(
            management_vec.reshape(
                (self.get_n(), 1)), self.get_m(), axis=1)
        self.Q[liq_ind:, :] *= management_matrix
        self.R[liq_ind:] += (self.P[liq_ind:] - new_p)
        self.P[liq_ind:] = new_p

    def record_defaults(self):
        self.cumdefaults_classic.append(self.classic_counter)
        self.cumdefaults_leverage.append(self.leverage_counter)

    def record_degrees(self):
        self.in_degrees.append(np.sum(self.get_nodes_indegree()))
        self.out_degrees.append(np.sum(self.get_nodes_outdegree()))

    def get_normalized_cumlosses(self):
        return np.cumsum(self.lost_value) / self.initial_value

    def snap_record(self):
        # rec_dic = dict()
        # rec_dic["L"] = self.L.copy()
        # rec_dic["L+"] = self.get_loans().copy()
        # rec_dic["D+"] = self.get_debts().copy()
        # rec_dic["R"] = self.R.copy()
        # rec_dic["Q"] = self.Q.copy()
        # rec_dic["P"] = self.P.copy()
        # rec_dic["E"] = self.get_equities()
        # rec_dic["Defaulting"] = self.get_defaulting()
        # rec_dic["Lost_value"] = self.lost_value[-1]
        # rec_dic["Out_degree"] = self.get_nodes_outdegree()
        # rec_dic["In_degree"] = self.get_nodes_indegree()
        # rec_dic["Defaulted"] = self.defaulted.copy()
        # self.record.append(rec_dic)
        return 0

    def get_equities_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        equities_tuple = tuple([self.record[t]["E"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(equities_tuple, axis=1)

    def get_reserves_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        reserves_tuple = tuple([self.record[t]["R"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(reserves_tuple, axis=1)

    def get_portfolios_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        portfolios_tuple = tuple([self.record[t]["P"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(portfolios_tuple, axis=1)

    def get_loans_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        loans_tuple = tuple(
            [self.record[t]["L+"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(loans_tuple, axis=1)

    def get_debts_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        debts_tuple = tuple(
            [self.record[t]["D+"].reshape((self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(debts_tuple, axis=1)

    def get_defaulting_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        defaulting_tuple = tuple([self.record[t]["Defaulting"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(defaulting_tuple, axis=1)

    def get_indegree_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        indegree_tuple = tuple([self.record[t]["In_degree"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(indegree_tuple, axis=1)

    def get_outdegree_record(self):
        T = len(self.record)
        liq_ind = int(self.liquidator)
        outdegree_tuple = tuple([self.record[t]["In_degree"].reshape(
            (self.get_n() + liq_ind, 1)) for t in range(0, T)])
        return np.concatenate(outdegree_tuple, axis=1)

    def get_defaults_cdf(self):
        defaulting = self.get_defaulting_record()
        cum_defaulting = np.cumsum(
            np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])]))
        return cum_defaulting
