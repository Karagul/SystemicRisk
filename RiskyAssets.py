import numpy as np
import pandas as pd


class GaussianAssets:

    def __init__(self, mus, sigmas, initvalues, length):
        self.mus = mus
        self.sigmas = sigmas
        self.length = length
        self.initvalues = initvalues

    def generate(self):
        gaussians = np.random.normal(self.mus, self.sigmas, (self.length, self.mus.shape[0]))
        returns_factors = 1 + gaussians
        log_factors = pd.DataFrame(data=returns_factors, dtype=np.float64).apply(np.log)
        cum_log_factors = log_factors.cumsum(axis=0)
        factors = cum_log_factors.apply(np.exp).as_matrix()
        return self.initvalues * factors



