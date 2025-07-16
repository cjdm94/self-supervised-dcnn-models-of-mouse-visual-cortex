import os
import pickle
import numpy as n


def process_neural_data():
    """
    taken from code_and_instructions/code_NCE_fig_3/natimg.py
    https://figshare.com/s/aac82eac1829b7ec406e?file=54722153
    """
    output_dir = './test-mouse-data'
    N_REPETITIONS_PER_STIM = 2

    os.makedirs(output_dir, exist_ok=True)

    for mouse_id in ["M01-D2", "M02-D3", "M03-D4"]:
        path_to_data = f'data/natimg-data-{mouse_id}.pickle'

        print(f"Loading data from {path_to_data}")
        with open(path_to_data, "rb") as f:
            data = pickle.load(f)

        stims_orig = data["stims"]
        resps_orig = data["resps"]

        # Initial data characterization (based on a fixed seed, as in notebook)
        # Seed for initial respmat construction and sigvar calculation
        n.random.seed(10)
        unique_stims_init = n.unique(stims_orig)
        n_trial_init, n_cell_init = resps_orig.shape
        n_rep_counts_init = n.zeros_like(unique_stims_init, dtype=int)
        for idx, unique_stim_val in enumerate(unique_stims_init):
            n_occ = len(n.where(stims_orig == unique_stim_val)[0])
            n_rep_counts_init[idx] = n_occ

        use_stims_init = unique_stims_init[n_rep_counts_init >=
                                           N_REPETITIONS_PER_STIM]
        n_stim_init = len(use_stims_init)
        print(n_stim_init)

        respmat_init = n.zeros(
            (n_stim_init, N_REPETITIONS_PER_STIM, n_cell_init))
        for idx, stim_val in enumerate(use_stims_init):
            stim_trials_indices = n.where(stims_orig == stim_val)[0]
            chosen_trials_indices = n.random.choice(
                stim_trials_indices, N_REPETITIONS_PER_STIM, replace=False
            )
            for rep_idx in range(N_REPETITIONS_PER_STIM):
                respmat_init[idx,
                             rep_idx] = resps_orig[chosen_trials_indices[rep_idx]]

        # Save each file to a subdir named after the mouse ID
        mouse_dir = os.path.join(output_dir, mouse_id.lower())
        os.makedirs(mouse_dir, exist_ok=True)

        n.save(os.path.join(mouse_dir, 'resps.npy'), respmat_init)
        n.save(os.path.join(mouse_dir, 'stims.npy'), use_stims_init)

        print(f"Shape of reshaped neural data: {respmat_init.shape}")
