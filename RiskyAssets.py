import numpy as np
import pandas as pd
from scipy import stats


class MultiplicativeGaussian:

    def __init__(self, mus, sigmas, initvalues, length):
        self.mus = mus
        self.sigmas = sigmas
        self.length = length
        self.initvalues = initvalues

    def generate(self):
        gaussians = np.random.normal(
            self.mus, self.sigmas, (self.length, self.mus.shape[0]))
        returns_factors = 1 + gaussians
        log_factors = pd.DataFrame(
            data=returns_factors,
            dtype=np.float64).apply(
            np.log)
        cum_log_factors = log_factors.cumsum(axis=0)
        factors = cum_log_factors.apply(np.exp).as_matrix()
        trajectories = self.initvalues * factors
        trajectories = np.insert(trajectories, [0], self.initvalues, axis=0)
        trajectories[trajectories < 0] = 0
        return trajectories


class AdditiveGaussian:

    def __init__(self, mus, sigmas, initvalues, length):
        self.mus = mus
        self.sigmas = sigmas
        self.length = length
        self.initvalues = initvalues

    def generate(self):
        m = self.mus.shape[0]
        gaussians = np.zeros((self.length, m))
        for i in range(0, m):
            gaussians[:, i] = np.cumsum(np.random.normal(
                self.mus[i], self.sigmas[i], self.length))
        repeated = np.repeat(
            self.initvalues.reshape(
                (1, m)), self.length, axis=0)
        trajectories = repeated + gaussians
        trajectories = np.insert(trajectories, [0], self.initvalues, axis=0)
        for i in range(0, m):
            stop_index = np.argwhere(trajectories[:, i] <= 0)
            if stop_index.shape[0] > 0:
                trajectories[stop_index[0][0]:, i] = 0
        return trajectories
