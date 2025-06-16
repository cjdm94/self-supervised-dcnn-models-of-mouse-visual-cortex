import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, CenterCrop
import torch
from torch.utils.data import TensorDataset
from torchvision import utils as torch_utils
from pathlib import Path
from cortexlib.utils.file import find_project_root


class CortexlabImages:
    def __init__(self, path_to_data=None, size=(96, 96)):
        if path_to_data is None:
            current_file = Path(__file__)
            project_root = find_project_root(current_file)
            path_to_data = project_root / 'data' / 'selection1866'
        else:
            path_to_data = Path(path_to_data).resolve()

        self.path_to_data = path_to_data

        # Resize shortest edge to 96 (cut off the rightmost part of the image)
        self.transform = Compose([
            Resize(size[0]),
            CenterCrop(size),
            Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def load_images_shown_to_mouse(self, mouse_image_ids):
        # e.g. we have 1866 images in the set, but the neural response data for the mouse
        # only uses 1573 of them because some ~300 images didn't have two repeats, so were disposed
        # therefore we filter the full set here so that we only use the relevant 1573
        stim_ids = mouse_image_ids.astype(int)
        # stim_ids = np.sort(mouse_image_ids.astype(int))
        tensors, labels = [], []

        for stim_id in stim_ids:
            img_np = self._load_mat_image(stim_id)
            img_tensor = self._preprocess_image(img_np)
            tensors.append(img_tensor)
            labels.append(stim_id)

        dataset = TensorDataset(torch.stack(tensors), torch.tensor(labels))
        return dataset

    def _load_mat_image(self, stim_id):
        filepath = os.path.join(self.path_to_data, f'img{stim_id}.mat')
        data = loadmat(filepath)
        # Take leftmost part of the image - each "image" is a trio of images
        img = data['img'][:, :500]
        return np.stack([img]*3, axis=-1)  # Convert to RGB

    def _preprocess_image(self, img_np):
        tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return self.transform(tensor)

    def show_sample(self, dataset, n=12):
        images, _ = dataset.tensors
        grid = torch_utils.make_grid(
            images[:n], nrow=6, normalize=True, pad_value=0.9)
        grid = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 5))
        plt.title('Processed images: sample')
        plt.imshow(grid)
        plt.axis('off')
        plt.show()

    def plot_raw_image(self, stim_id):
        img_np = self._load_mat_image(stim_id)[:, :, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(img_np, cmap='gray')
        plt.colorbar(label='Pixel Intensity')
        plt.title(f"Raw Image {stim_id}")
        plt.axis('off')
        plt.show()
