import numpy as np

class GaussianAssets:

    def __init__(self, mus, sigmas, length):
        self.mus = mus
        self.sigmas = sigmas
        self.length = length

    def generate(self):
        gaussians = np.random.normal(self.mus, self.sigmas, (self.length, self.mus.shape[0]))



