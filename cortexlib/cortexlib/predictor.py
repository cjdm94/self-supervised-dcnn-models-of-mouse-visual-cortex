from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA


class NeuralResponsePredictor:
    """
    A class to predict neural responses using model features.
    This class prepares the data, computes R-squared values, and calculates
    the Fraction of Explained Variance (FEV) for the predictions.
    It assumes that the input images_representation is a 2D tensor (flattened)
    or a 4D tensor (e.g., from intermediate layers of a SimCLR model).
    The neural_responses should be a 3D tensor with shape (samples, trials, neurons).
    """

    def __init__(self, predicting_neural_pc=False, neural_data_pc_index=None):
        if predicting_neural_pc is True and neural_data_pc_index is None:
            raise ValueError(
                "pc_index must be set when use_pca_on_neural_data is True")

        self.predicting_neural_pc = predicting_neural_pc
        self.pc_index = neural_data_pc_index

    def _prepare_data(self, images_representation, neural_responses_averaged_across_trials):
        """
        Prepares the data for regression by splitting it into training and test sets.

        Args:
            images_representation: 2D or 4D tensor of shape (num_images, num_features) or (num_images, channels, height, width).
            neural_responses_averaged_across_trials: 2D tensor of shape (num_images, num_neurons), where each image has a response averaged across trials.

        Returns:
            A dictionary containing:
                - x_train: Training features (2D tensor).
                - x_test: Test features (2D tensor).
                - y_train: Training labels - prediction targets (2D tensor).
                - y_test: Test labels - prediction targets (2D tensor).
        """
        # Ensure images_representation is a 2D tensor
        # e.g. SimCLR intermediate layers are 4D - e.g. (images, channels, height, width)
        X = images_representation
        if X.ndim > 2:
            X = X.view(X.shape[0], -1)

        Y = neural_responses_averaged_across_trials

        # FEV = 1 - (MSE - noise_var) / explainable_var
        # MSE is computed on test predictions (test_pred vs. y_test)
        # Explainable variance is total variance (in y_test) minus noise variance
        # Thus noise variance must be computed using *only the test images*
        # So, split into train/test while keeping the indexes, so that they can be passed to FEV computer later
        num_samples = X.shape[0]
        all_indices = np.arange(num_samples)

        train_indices, test_indexes = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        x_train, x_test = X[train_indices], X[test_indexes]
        y_train, y_test = Y[train_indices], Y[test_indexes]

        return {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'test_indexes': test_indexes
        }

    def _apply_pca_to_neural_data(self, y_train, y_test):
        """
        Applies PCA to the neural responses to reduce dimensionality.

        Args:
            y_train: 2D tensor of shape (num_train_images, num_neurons); this is the training subset of neural responses averaged across trials.
            y_test: 2D tensor of shape (num_test_images, num_neurons); this is the test subset of neural responses averaged across trials.

        Returns:
            A dictionary containing:
                - y_train: 1D tensor of shape (num_train_images,); the selected principal component from the training subset.
                - y_test: 1D tensor of shape (num_test_images,); the selected principal component from the test subset.        
        """
        pca = PCA(n_components=self.pc_index + 1)
        # fit_transform computes PCA on the training data
        y_train_pcs = pca.fit_transform(y_train)
        # transform applies the same PCA to the test data
        y_test_pcs = pca.transform(y_test)
        y_train = y_train_pcs[:, self.pc_index]
        y_test = y_test_pcs[:, self.pc_index]

        return {
            'y_train': y_train,
            'y_test': y_test
        }

    def _fit_regression_model(self, x_train, x_test, y_train, y_test):
        """
        Fits a Ridge regression model to the training data and evaluates it on the test data.

        Args:
            x_train: 2D tensor of shape (num_train_images, num_features).
            x_test: 2D tensor of shape (num_test_images, num_features).
            y_train: 2D tensor of shape (num_train_images, num_neurons).
            y_test: 2D tensor of shape (num_test_images, num_neurons).

        Returns:
            A dictionary containing:
                - test_pred: Predicted neural responses for the test subset.
                - train_pred: Predicted neural responses for the training subset.
                - test_r2: R-squared value for the test subset.
                - train_r2: R-squared value for the training subset.
        """
        alphas = np.logspace(-6, 6, 13)
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(x_train, y_train)

        train_pred = ridge.predict(x_train)
        test_pred = ridge.predict(x_test)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        return {
            'test_pred': test_pred,
            'train_pred': train_pred,
            'test_r2': test_r2,
            'train_r2': train_r2
        }

    def _compute_fev(self, y_true, y_pred, neural_responses):
        """
        Computes the Fraction of Explained Variance (FEV) for the test predictions.

        Args:
            y_true: 2D tensor of shape (num_test_images, num_neurons); this is the test subset of neural responses averaged across trials.
            y_pred: 2D tensor of shape (num_test_images, num_neurons); this is the predicted neural responses for the test subset.
            neural_responses: 3D tensor of shape (num_images, num_trials, num_neurons); this is the test subset of the original neural responses for *all trials*,
            from which y_true and y_pred were derived.

        Returns:
            A dictionary containing:
                - fev: Fraction of Explained Variance for each neuron.
                - mean_fev: Mean FEV across all neurons.
                - fev_filtered: Filtered FEV values for neurons with explainable variance above a threshold.
                - mean_fev_filtered: Mean filtered FEV across selected neurons.
        """
        # note: when computing variance from a sample (e.g. a subset of trials), using ddof=1 gives an unbiased estimate of the true population variance

        # variance across *images* - already averaged across trials (2 presentations per image)
        total_var = np.var(y_true, axis=0, ddof=1)

        # variance across *trials* - original shape: (1573, 2, 500)
        trial_var = np.var(
            neural_responses, axis=1, ddof=1)

        noise_var = np.mean(trial_var, axis=0)

        mse_test = mean_squared_error(y_true, y_pred, multioutput='raw_values')

        explainable_var = total_var - noise_var

        fev = 1 - (mse_test - noise_var) / explainable_var
        fev = np.clip(fev, 0, 1)

        # Filter out neurons with very low explainable variance
        # so that FEV is not dominated by noise
        fev_filtered = fev[explainable_var > 1e-3]

        return {
            'fev': fev,
            'mean_fev': np.mean(fev),
            'fev_filtered': fev_filtered,
            'mean_fev_filtered': np.mean(fev_filtered)
        }

    def compute_r_squared(self, images_representation, neural_responses):
        """
        Computes R-squared values for the given images representation and neural responses.

        Args:
            images_representation: 2D or 4D tensor of shape (num_images, num_features) or (num_images, channels, height, width).
            neural_responses: 3D tensor of shape (num_images, num_trials, num_neurons), where each image has multiple trials (e.g., 2).

        Returns:
            A dictionary containing:
                - test_pred: Predicted neural responses for the test subset.
                - train_pred: Predicted neural responses for the training subset.
                - test_r2: R-squared value for the test subset.
                - train_r2: R-squared value for the training subset.
        """
        # Average across trials, shape: (num_images, num_neurons)
        neural_responses_averaged_across_trials = neural_responses.mean(axis=1)

        d = self._prepare_data(
            images_representation, neural_responses_averaged_across_trials)

        y_train, y_test = d['y_train'], d['y_test']
        if self.predicting_neural_pc is True:
            pca = self._apply_pca_to_neural_data(y_train, y_test)
            y_train, y_test = pca['y_train'], pca['y_test']

        return self._fit_regression_model(
            d['x_train'], d['x_test'], y_train, y_test)

    def compute_fev(self, images_representation, neural_responses):
        """
        Computes the Fraction of Explained Variance (FEV) for the given images representation and neural responses.

        Args:
            images_representation: 2D or 4D tensor of shape (num_images, num_features) or (num_images, channels, height, width).
            neural_responses: 3D tensor of shape (num_images, num_trials, num_neurons), where each image has multiple trials (e.g., 2).

        Returns:
            A dictionary containing:
                - fev: Fraction of Explained Variance for each neuron.
                - mean_fev: Mean FEV across all neurons.
                - fev_filtered: Filtered FEV values for neurons with explainable variance above a threshold.
                - mean_fev_filtered: Mean filtered FEV across selected neurons.
        """
        # Average across trials, shape: (num_images, num_neurons)
        neural_responses_averaged_across_trials = neural_responses.mean(axis=1)

        d = self._prepare_data(
            images_representation, neural_responses_averaged_across_trials)

        y_train, y_test = d['y_train'], d['y_test']
        if self.predicting_neural_pc is True:
            pca = self._apply_pca_to_neural_data(y_train, y_test)
            y_train, y_test = pca['y_train'], pca['y_test']

        m = self._fit_regression_model(
            d['x_train'], d['x_test'], y_train, y_test)

        # use the test indexes: fev cares *only about neural responses to the test images*
        # used for the regression model, from which y_test and test_pred were derived
        neural_responses = neural_responses[d['test_indexes'], :, :]

        return self._compute_fev(y_test, m['test_pred'], neural_responses)
