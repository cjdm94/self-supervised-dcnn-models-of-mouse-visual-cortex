import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
import random
from cortexlib.utils.random import GLOBAL_SEED


class RSA:
    def __init__(self, neural_data, neural_data_pc_index=None, seed=GLOBAL_SEED):
        # Set global seeds for full determinism
        np.random.seed(seed)
        random.seed(seed)

        self.neural_data_pc_index = neural_data_pc_index
        self.vec_neural = self._prepare_neural_vector(
            neural_data, neural_data_pc_index)

    def _prepare_neural_vector(self, neural_data, neural_data_pc_index=None):
        if neural_data_pc_index is None:
            neural_data_prepared = neural_data
            metric = 'correlation'
        else:
            pca = PCA(n_components=neural_data_pc_index + 1)
            pcs = pca.fit_transform(neural_data)
            # only the requested PC
            neural_data_prepared = pcs[:, [neural_data_pc_index]]
            # if taking a single PC, correlation is undefined
            metric = 'euclidean'

        rdm = self._compute_rdm(neural_data_prepared, metric)
        return self._vectorise_rdm(rdm)

    @staticmethod
    def _compute_rdm(X, metric='correlation'):
        return squareform(pdist(X, metric=metric))

    @staticmethod
    def _vectorise_rdm(rdm):
        triu_idx = np.triu_indices(rdm.shape[0], k=1)
        return rdm[triu_idx]

    @staticmethod
    def _stable_spearman(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) < 2:
            return np.nan
        r1 = rankdata(a[mask])
        r2 = rankdata(b[mask])
        if np.std(r1) == 0 or np.std(r2) == 0:
            return np.nan
        return np.corrcoef(r1, r2)[0, 1]

    def compute_similarity(self, images_feats):
        rdm = self._compute_rdm(images_feats)
        vec_feats = self._vectorise_rdm(rdm)

        if self.neural_data_pc_index is None:
            # If using neural data with original dimensionality, use the default correlation metric
            return spearmanr(self.vec_neural, vec_feats).correlation
        else:
            # If taking a single PC of neural data, use the stable Spearman correlation
            return self._stable_spearman(self.vec_neural, vec_feats)
