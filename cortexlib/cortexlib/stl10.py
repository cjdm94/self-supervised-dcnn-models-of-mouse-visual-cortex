import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import ConcatDataset, TensorDataset
import torch
import random
from typing import List
from pathlib import Path
from cortexlib.utils.file import find_project_root
from cortexlib.utils.random import GLOBAL_SEED
import numpy as np


class STL10FewShotDataset:
    def __init__(self, n_per_class: int = 100, image_size: tuple = (96, 96), seed: int = GLOBAL_SEED, data_root=None):
        # Set global seeds for full determinism
        np.random.seed(seed)
        random.seed(seed)

        if data_root is None:
            current_file = Path(__file__)
            project_root = find_project_root(current_file)
            data_root = project_root / 'data'
        else:
            data_root = Path(data_root).resolve()

        self.data_root = data_root
        self.class_names_file = f'{self.data_root}/stl10_binary/{STL10.class_names_file}'

        self.n_per_class = n_per_class
        self.image_size = image_size
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        self._load_datasets()
        self._select_n_per_class()
        self._load_class_names()

    def _load_datasets(self):
        train_set = STL10(root=self.data_root, split='train',
                          download=True, transform=self.transform)
        test_set = STL10(root=self.data_root, split='test',
                         download=True, transform=self.transform)
        self.labeled = ConcatDataset([train_set, test_set])

    def _select_n_per_class(self):
        label_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.labeled):
            if len(label_to_indices[label]) < self.n_per_class:
                label_to_indices[label].append(idx)
            if all(len(v) == self.n_per_class for v in label_to_indices.values()):
                break

        selected_indices = [i for indices in label_to_indices.values()
                            for i in indices]
        images = torch.stack([self.labeled[i][0] for i in selected_indices])
        labels = torch.tensor([self.labeled[i][1] for i in selected_indices])
        self.dataset = TensorDataset(images, labels)
        self.images = images
        self.labels = labels

    def _load_class_names(self):
        with open(file=self.class_names_file, mode='r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def get_dataset(self) -> TensorDataset:
        return self.dataset

    def get_class_names(self) -> List[str]:
        return self.class_names

    def stats(self) -> None:
        print("Shape of images:", self.images.shape)
        print("Shape of labels:", self.labels.shape)
        print("Unique labels:", self.labels.unique())
        print("Number of images:", len(self.dataset))
        print("Class names:", self.class_names)
