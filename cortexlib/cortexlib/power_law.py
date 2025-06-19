from cortexlib.utils.random import GLOBAL_SEED
from sklearn.decomposition import PCA
import powerlaw
import numpy as np
import random


class PowerLawAlphaEstimator:
    def __init__(self, verbose=False, seed=GLOBAL_SEED):
        self.verbose = verbose
        self.seed = seed

        # Set global seeds for full determinism
        np.random.seed(seed)
        random.seed(seed)

    def compute_alpha(self, feats):
        pca = PCA(n_components=min(feats.shape),
                  random_state=self.seed, svd_solver='full')

        pca.fit(feats)
        explained_var = pca.explained_variance_ratio_

        fit = powerlaw.Fit(
            explained_var, verbose=self.verbose, random_state=self.seed)
        alpha = fit.power_law.alpha

        fit_no_pc1 = powerlaw.Fit(explained_var[1:], verbose=self.verbose)
        alpha_no_pc1 = fit_no_pc1.power_law.alpha

        return {
            'alpha': alpha,
            'alpha_no_pc1': alpha_no_pc1
        }
