import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA


@dataclass
class AlphaResult:
    alpha: float
    alpha_no_pc1: float
    kmin: int
    kmax: int
    r2: float


class PowerLawAlphaEstimator:
    def __init__(self, verbose=False, seed=0):
        self.verbose = verbose
        self.seed = seed
        np.random.seed(seed)

    # --------- helpers ---------
    @staticmethod
    def _select_window(length, kmin, kmax):
        kmax = int(min(kmax, length))
        kmin = int(max(1, kmin))
        if kmax <= kmin:
            raise ValueError(f"kmax ({kmax}) must be > kmin ({kmin}).")
        return kmin, kmax

    @staticmethod
    def _logspace_sample_indices(kmin, kmax, npts=100):
        # real-valued log-spaced positions, then round & uniquify
        ks = np.exp(np.linspace(np.log(kmin), np.log(kmax), int(npts)))
        idx = np.unique(np.clip(np.round(ks).astype(int), kmin, kmax))
        return idx

    @staticmethod
    def _ols_loglog(x_ranks, y_vals):
        # log–log regression: log(y) = a + b*log(x)
        xlog = np.log(x_ranks)
        ylog = np.log(y_vals)
        A = np.vstack([xlog, np.ones_like(xlog)]).T
        b, a = np.linalg.lstsq(A, ylog, rcond=None)[0]  # slope, intercept
        # r^2 as squared Pearson correlation
        r = np.corrcoef(xlog, ylog)[0, 1]
        r2 = float(r * r)
        alpha = float(-b)
        return alpha, -b, a, r2  # alpha, slope, intercept, r2

    @staticmethod
    def _alpha_from_eigs(eigs, kmin=11, kmax=500, log_spaced=False, npts=100):
        kmin, kmax = PowerLawAlphaEstimator._select_window(
            len(eigs), kmin, kmax)

        if log_spaced:
            ks = PowerLawAlphaEstimator._logspace_sample_indices(
                kmin, kmax, npts=npts)
        else:
            ks = np.arange(kmin, kmax + 1, dtype=int)

        y = np.asarray(eigs, dtype=float)
        vals = y[ks - 1]  # ranks are 1-indexed

        # guard against non-positive due to numerical noise
        mask = vals > 0
        ks = ks[mask]
        vals = vals[mask]

        alpha, slope, intercept, r2 = PowerLawAlphaEstimator._ols_loglog(
            ks, vals)

        # alpha without PC1 (i.e., start at k=2 within the chosen window)
        # we do this by discarding rank 1 if present
        if len(ks) >= 2 and ks[0] == 1:
            ks_no1 = ks[1:]
            vals_no1 = vals[1:]
        else:
            # if our window already starts at >=2 (e.g., kmin=11), this equals alpha
            ks_no1 = ks
            vals_no1 = vals

        alpha_no_pc1, *_ = PowerLawAlphaEstimator._ols_loglog(ks_no1, vals_no1)

        return AlphaResult(alpha=alpha, alpha_no_pc1=alpha_no_pc1,
                           kmin=int(ks[0]), kmax=int(ks[-1]), r2=r2)

    # --------- user-facing API ---------
    def compute_alpha_rank(self, feats, kmin=11, kmax=500,
                           log_spaced=False, npts=100):
        """
        PCA on a single-response matrix; fit power-law on eigen-variances.
        feats: shape (n_stimuli, n_neurons) — rows are stimuli.
        """
        pca = PCA(n_components=min(feats.shape),
                  random_state=self.seed, svd_solver='full')
        pca.fit(feats)
        # Use explained_variance_ (variance units), not ratio
        eigs = pca.explained_variance_
        res = self._alpha_from_eigs(eigs, kmin=kmin, kmax=kmax,
                                    log_spaced=log_spaced, npts=npts)
        if self.verbose:
            print(f"[alpha_rank] α={res.alpha:.3f} (k={res.kmin}..{res.kmax}, "
                  f"log_spaced={log_spaced}, r²={res.r2:.3f})")
        return res

    def compute_cv_alpha_rank(self, rep1, rep2, kmin=11, kmax=500,
                              log_spaced=False, npts=100):
        """
        cvPCA spectrum then power-law fit:
          1) SVD on repeat 1 (stimuli × neurons).
          2) Project both repeats onto V (PC axes from repeat 1).
          3) Signal eigenvalues = diag( proj1.T @ proj2 ) / n_stimuli.
        rep1, rep2: arrays shape (n_stimuli, n_neurons), same stimuli order.
        """
        X1 = rep1 - rep1.mean(axis=0, keepdims=True)
        X2 = rep2 - rep2.mean(axis=0, keepdims=True)

        # PCs from repeat 1
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        proj1 = X1 @ Vt.T
        proj2 = X2 @ Vt.T

        Ns = X1.shape[0]
        C = (proj1.T @ proj2) / float(Ns)
        eigs_signal = np.abs(np.diag(C))  # guard tiny negatives

        res = self._alpha_from_eigs(eigs_signal, kmin=kmin, kmax=kmax,
                                    log_spaced=log_spaced, npts=npts)
        if self.verbose:
            print(f"[cv_alpha_rank] α={res.alpha:.3f} (k={res.kmin}..{res.kmax}, "
                  f"log_spaced={log_spaced}, r²={res.r2:.3f})")
        return res
