from sklearn.decomposition import PCA
import powerlaw


class PowerLawAlphaEstimator:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute_alpha(self, feats):
        pca = PCA(n_components=min(feats.shape))
        pca.fit(feats)
        explained_var = pca.explained_variance_ratio_

        fit = powerlaw.Fit(explained_var, verbose=self.verbose)
        alpha = fit.power_law.alpha

        fit_no_pc1 = powerlaw.Fit(explained_var[1:], verbose=self.verbose)
        alpha_no_pc1 = fit_no_pc1.power_law.alpha

        return {
            'alpha': alpha,
            'alpha_no_pc1': alpha_no_pc1
        }
