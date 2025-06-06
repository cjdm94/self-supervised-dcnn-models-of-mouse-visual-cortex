import numpy as np
from os import path
import matplotlib.pyplot as plt


class CortexlabMouse:
    def __init__(self, path_to_data='../../data/neural'):
        self.path_to_data = path_to_data
        self.neural_responses = None
        self.stimulus_ids = None
        self._load_data()

    def compute_null_all_neurons(self, n_shuffles=100):
        # imresps shape = (1573, 2, 15363)
        # responses in imresps shape = (2, 15363)
        num_stimuli = self.neural_responses.shape[0]  # 1573
        num_repeats = self.neural_responses.shape[1]  # 2
        num_neurons = self.neural_responses.shape[2]  # 15363

        null_srv_all_neurons = []  # shape (n_shuffles, num_neurons)

        for _ in range(n_shuffles):
            # Shuffle stimulus indices *twice* to create two independent splits!
            shuffled_indices_A = np.random.permutation(num_stimuli)
            shuffled_indices_B = np.random.permutation(num_stimuli)

            # Now for the splits, we can just use fixed repeat indices,
            # because for each split, at index N the responses correspond to different stimuli
            # e.g. split_A = [ stim_100_repeat_1, stim_2_repeat_1, stim_19_repeat_1, ... ]
            # e.g. split_B = [ stim_543_repeat_2, stim_345_repeat_2, stim_3_repeat_2, ... ]
            split_A = self.neural_responses[shuffled_indices_A, 0, :]
            split_B = self.neural_responses[shuffled_indices_B, 1, :]

            # Compute SRV for the shuffled data
            fraction_of_stimulus_variance, _ = self._compute_signal_related_variance(
                split_A, split_B)
            null_srv_all_neurons.append(fraction_of_stimulus_variance)

        null_srv_all_neurons = np.array(null_srv_all_neurons)
        null_srv_all_neurons.shape  # (100, 15363)

        print(null_srv_all_neurons[0])
        print(null_srv_all_neurons[33])

        return null_srv_all_neurons

    def compute_real_srv_all_neurons(self):
        split_A, split_B = [], []
        # responses shape: (2, n_neurons)
        for responses in self.neural_responses:
            indices = np.random.permutation(2)  # Randomly shuffle [0, 1]
            # Assign one repeat to split_A
            split_A.append(responses[indices[0]])
            # Assign the other to split_B
            split_B.append(responses[indices[1]])

        split_A = np.array(split_A)  # Shape: (n_stimuli, n_neurons)
        split_B = np.array(split_B)  # Shape: (n_stimuli, n_neurons)

        # Compute SRV for real data
        real_srv_all_neurons, stim_to_noise_ratio = self._compute_signal_related_variance(
            split_A, split_B)

        print(real_srv_all_neurons)
        print(stim_to_noise_ratio)

        # Should be (15363,)
        print("Real SRV shape:", real_srv_all_neurons.shape)

        return real_srv_all_neurons

    def get_reliable_neuron_indices(self, null_srv_all_neurons, real_srv_all_neurons, percentile_threshold=99):
        # This gives the Nth-percentile SRV value of the null distribution for each neuron
        # In other words the threshold for each neuron to be considered reliable
        # e.g. if neuron 0 has a null distribution of SRVs across 10 shuffles
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], the threshold would be 0.9
        top_nth_percentile_null = np.percentile(
            null_srv_all_neurons, percentile_threshold, axis=0)
        # [0.03651716 0.03126347 0.03325775 ... 0.02738261 0.03546677 0.0333109 ]
        print(top_nth_percentile_null)

        # Get indices of reliable neurons
        reliable_neuron_indices = np.where(
            real_srv_all_neurons >= top_nth_percentile_null)[0]

        print(f"Number of reliable neurons: {len(reliable_neuron_indices)}")
        print(f"Indices of reliable neurons: {reliable_neuron_indices}")

        return reliable_neuron_indices

    def get_responses_for_reliable_neurons(self, reliable_neuron_indices, real_srv_all_neurons, num_neurons=500):
        """
        Filter the neural responses to only include the reliable neurons.
        """
        if len(reliable_neuron_indices) < num_neurons:
            print(f"Warning: Found only {len(reliable_neuron_indices)} reliable neurons, "
                  f"which is less than the requested {num_neurons}. Adjusting to available neurons.")
            num_neurons = len(reliable_neuron_indices)

        reliable_srv_scores = real_srv_all_neurons[reliable_neuron_indices]
        sorted_indices = np.argsort(reliable_srv_scores)[::-1]
        most_reliable_neurons = reliable_neuron_indices[sorted_indices[:num_neurons]]
        highest_srv_scores = real_srv_all_neurons[most_reliable_neurons]
        neural_responses = self.neural_responses[:, :, most_reliable_neurons]
        neural_responses_mean = neural_responses.mean(axis=1)

        assert most_reliable_neurons.shape[0] == num_neurons, "Mismatch in neuron selection!"
        print("Dimensionality of neural responses:",
              neural_responses_mean.shape)
        print("Top 500 reliable neuron indices:", most_reliable_neurons[:10])
        print("Corresponding SRV scores:", highest_srv_scores[:10])
        print("Top 500 neural responses shape:",
              neural_responses.shape)  # (1573, 2, 500)
        print("Averaged top 500 neural responses shape:",
              neural_responses_mean.shape)  # (1573, 500)

        return neural_responses_mean, neural_responses, most_reliable_neurons

    def plot_null_distribution_for_neuron(self, null_srv_all_neurons, neuron_index=0):
        plt.hist([srv[neuron_index] for srv in null_srv_all_neurons],
                 bins=100, color='blue', alpha=0.7)
        plt.xlabel("Fraction of Stimulus-Related Variance (SRV)")
        plt.ylabel("Number of Shuffles")
        plt.title(f"Null Distribution of SRV for Neuron {neuron_index}")
        plt.show()

    def plot_real_srv_distribution(self, real_srv_all_neurons, reliable_indices):
        plt.hist(real_srv_all_neurons, bins=100, color='red', alpha=0.7)
        plt.hist(real_srv_all_neurons[reliable_indices],
                 bins=100, color='blue', alpha=0.7)
        plt.xlabel("Fraction of Stimulus-Related Variance (SRV)")
        plt.ylabel("Number of Neurons")
        plt.title("All Neurons: SRV all vs. SRV reliable")
        plt.show()

        plt.hist(real_srv_all_neurons[reliable_indices],
                 bins=100, color='blue', alpha=0.7)
        plt.xlabel("Fraction of Stimulus-Related Variance (SRV)")
        plt.ylabel("Number of Neurons")
        plt.title("SRV Distribution for Reliable Neurons")
        plt.show()

    def _load_data(self):
        """
        Load neural response data and stimulus IDs from the specified path.

        imresps.npy is of shape (1573, 2, 15363), where 1573 is number of images, 2 repeats each, and 15363 neurons recorded
        stimids.npy has the image id (matching the image dataset ~selection1866~) for each stimulus number,
        so of you want to see what image was presented on imresps[502] you would check stim_ids[502]
        """
        self.neural_responses = np.load(
            path.join(self.path_to_data, 'imresps.npy'))
        self.stimulus_ids = np.load(
            path.join(self.path_to_data, 'stimids.npy'))

        print(f"Loaded imresps with shape: {self.neural_responses.shape}")
        print(f"Loaded stimids with shape: {self.stimulus_ids.shape}")

    def _compute_signal_related_variance(self, resp_a, resp_b, mean_center=True):
        """
        compute the fraction of signal-related variance for each neuron,
        as per Stringer et al Nature 2019. Cross-validated by splitting
        responses into two halves. Note, this only is "correct" if resp_a
        and resp_b are *not* averages of many trials.

        Args:
            resp_a (ndarray): n_stimuli, n_cells
            resp_b (ndarray): n_stimuli, n_cells

        Returns:
            fraction_of_stimulus_variance: 0-1, 0 is non-stimulus-caring, 1 is only-stimulus-caring neurons
            stim_to_noise_ratio: ratio of the stim-related variance to all other variance
        """
        if len(resp_a.shape) > 2:
            # if the stimulus is multi-dimensional, flatten across all stimuli
            resp_a = resp_a.reshape(-1, resp_a.shape[-1])
            resp_b = resp_b.reshape(-1, resp_b.shape[-1])
        ns, nc = resp_a.shape
        if mean_center:
            # mean-center the activity of each cell
            resp_a = resp_a - resp_a.mean(axis=0)
            resp_b = resp_b - resp_b.mean(axis=0)

        # compute the cross-trial stimulus covariance of each cell
        # dot-product each cell's (n_stim, ) vector from one half
        # with its own (n_stim, ) vector on the other half

        covariance = (resp_a * resp_b).sum(axis=0) / ns

        # compute the variance of each cell across both halves
        resp_a_variance = (resp_a**2).sum(axis=0) / ns
        resp_b_variance = (resp_b**2).sum(axis=0) / ns
        total_variance = (resp_a_variance + resp_b_variance) / 2

        if np.any(total_variance < 1e-12):
            print(
                f"Warning: Near-zero total variance for neurons: {np.where(total_variance < 1e-12)[0]}")

        # compute the fraction of the total variance that is
        # captured in the covariance
        fraction_of_stimulus_variance = covariance / total_variance

        # if you want, you can compute SNR as well:
        stim_to_noise_ratio = fraction_of_stimulus_variance / (
            1 - fraction_of_stimulus_variance
        )

        return fraction_of_stimulus_variance, stim_to_noise_ratio
