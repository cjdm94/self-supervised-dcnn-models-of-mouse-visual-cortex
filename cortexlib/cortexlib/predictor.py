from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


class SimCLRNeuralPredictor:
    """
    A class to predict neural responses using SimCLR features.
    This class prepares the data, computes R-squared values, and calculates
    the Fraction of Explained Variance (FEV) for the predictions.
    It assumes that the input images_representation is a 2D tensor (flattened)
    or a 4D tensor (e.g., from intermediate layers of a SimCLR model).
    The neural_responses should be a 3D tensor with shape (samples, trials, neurons).
    """

    def _prepare_data(self, images_representation, neural_responses):
        """
        Prepares the data for regression by splitting it into training and test sets.

        Args:
            images_representation: 2D or 4D tensor of shape (num_images, num_features) or (num_images, channels, height, width).
            neural_responses: 3D tensor of shape (num_images, num_trials, num_neurons), where each image has multiple trials (e.g., 2).

        Returns: 
            A dictionary containing:
                - X_train: Training features (2D tensor).
                - X_test: Test features (2D tensor).
                - Y_train: Training labels - prediction targets (2D tensor).
                - Y_test: Test labels - prediction targets (2D tensor).
        """
        # Ensure images_representation is a 2D tensor
        # SimCLR intermediate layers are 4D - e.g. (images, channels, height, width)
        X = images_representation
        if X.ndim > 2:
            X = X.view(X.shape[0], -1)

        # Average across trials, shape: (num_images, num_neurons)
        Y = neural_responses.mean(axis=1)

        # FEV = 1 - (MSE - noise_var) / explainable_var
        # MSE is computed on test predictions (test_pred vs. Y_test)
        # Explainable variance is total variance (in Y_test) minus noise variance
        # Thus noise variance must be computed using only the test images
        # So, split into train/test while keeping the indices, so that they can be passed to FEV computer later
        num_samples = X.shape[0]
        all_indices = np.arange(num_samples)

        train_indices, test_indexes = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        X_train, X_test = X[train_indices], X[test_indexes]
        Y_train, Y_test = Y[train_indices], Y[test_indexes]

        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'test_indexes': test_indexes
        }

    def _fit_regression_model(self, X_train, X_test, Y_train, Y_test):
        """
        Fits a Ridge regression model to the training data and evaluates it on the test data.

        Args:
            X_train: 2D tensor of shape (num_train_images, num_features).
            X_test: 2D tensor of shape (num_test_images, num_features).
            Y_train: 2D tensor of shape (num_train_images, num_neurons).
            Y_test: 2D tensor of shape (num_test_images, num_neurons). 

        Returns:
            A dictionary containing:    
                - test_pred: Predicted neural responses for the test subset.
                - train_pred: Predicted neural responses for the training subset.
                - test_r2: R-squared value for the test subset.
                - train_r2: R-squared value for the training subset.        
        """
        alphas = np.logspace(-6, 6, 13)
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_train, Y_train)

        train_pred = ridge.predict(X_train)
        test_pred = ridge.predict(X_test)
        train_r2 = r2_score(Y_train, train_pred)
        test_r2 = r2_score(Y_test, test_pred)

        return {
            'test_pred': test_pred,
            'train_pred': train_pred,
            'test_r2': test_r2,
            'train_r2': train_r2
        }

    def _compute_fev(self, Y_test, test_pred, neural_responses):
        """
        Computes the Fraction of Explained Variance (FEV) for the test predictions.

        Args:
            Y_test: 2D tensor of shape (num_test_images, num_neurons); this is the test subset of neural responses averaged across trials.
            test_pred: 2D tensor of shape (num_test_images, num_neurons); this is the predicted neural responses for the test subset.
            neural_responses: 3D tensor of shape (num_images, num_trials, num_neurons); this is the test subset of the original neural responses for *all trials*, 
            from which Y_test and test_pred were derived.

        Returns:
            A dictionary containing:
                - fev: Fraction of Explained Variance for each neuron.
                - mean_fev: Mean FEV across all neurons.
                - fev_filtered: Filtered FEV values for neurons with explainable variance above a threshold.
                - mean_fev_filtered: Mean filtered FEV across selected neurons. 
        """
        # note: when computing variance from a sample (e.g. a subset of trials), using ddof=1 gives an unbiased estimate of the true population variance

        # variance across *images* - already averaged across trials (2 presentations per image)
        total_var_test = np.var(Y_test, axis=0, ddof=1)

        # variance across *trials* - original shape: (1573, 2, 500)
        trial_var_test = np.var(
            neural_responses, axis=1, ddof=1)

        # mean noise variance across *images*
        noise_var_test = np.mean(trial_var_test, axis=0)

        mse_test = mean_squared_error(
            Y_test, test_pred, multioutput='raw_values')

        explainable_var_test = total_var_test - noise_var_test

        fev = 1 - (mse_test - noise_var_test) / explainable_var_test
        fev = np.clip(fev, 0, 1)

        # Filter out neurons with very low explainable variance
        # so that FEV is not dominated by noise
        fev_filtered = fev[explainable_var_test > 1e-3]

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
        d = self._prepare_data(
            images_representation, neural_responses)

        return self._fit_regression_model(
            d['X_train'], d['X_test'], d['Y_train'], d['Y_test'])

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
        d = self._prepare_data(
            images_representation, neural_responses)

        m = self._fit_regression_model(
            d['X_train'], d['X_test'], d['Y_train'], d['Y_test'])

        # use the test indexes: fev cares *only about neural responses to the test images*
        # used for the regression model, from which Y_test and test_pred were derived
        neural_responses = neural_responses[d['test_indexes'], :, :]

        return self._compute_fev(d['Y_test'], m['test_pred'], neural_responses)
