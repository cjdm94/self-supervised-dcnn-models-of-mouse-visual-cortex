import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from enum import Enum

VGG_LAYER_NAMES = {
    "conv1_1": 0,
    "conv1_2": 2,
    "conv2_1": 5,
    "conv2_2": 7,
    "conv3_1": 10,
    "conv3_2": 12,
    "conv3_3": 14,
    "conv3_4": 16,
    "conv4_1": 19,
    "conv4_2": 21,
    "conv4_3": 23,
    "conv4_4": 25,
    "conv5_1": 28,
    "conv5_2": 30,
    "conv5_3": 32,
    "conv5_4": 34,
}


class PoolingMode(Enum):
    FLATTEN = "flatten"
    AVGPOOL = "avgpool"
    AVGPOOL7X7 = "avgpool7x7"


class PreTrainedVGG19Model:
    def __init__(self, layers_to_capture=None, batch_size=16, num_workers=4, device=None, pooling_mode=PoolingMode.FLATTEN):
        # ImageNet training settings
        self.training_images_size = (224, 224)
        self.training_images_normalise_mean = [0.485, 0.456, 0.406]
        self.training_images_normalise_std = [0.229, 0.224, 0.225]
        self.training_images_channels = 3  # RGB images
        self.training_images_rescale_per_image = False

        self.pooling_mode = pooling_mode
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.vgg19(
            pretrained=True).features.to(self.device).eval()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Default to conv3_1 if nothing passed
        self.layers_to_capture = layers_to_capture or {
            "conv3_1": VGG_LAYER_NAMES["conv3_1"]
        }

        self.activations = {}
        self._register_hooks()

    def get_image_settings(self):
        """
        Returns the settings for the training images used in the VGG-19 model.
        Any images passed to the model should be preprocessed to match these settings.

        Returns:
            dict: A dictionary containing the resolution, number of channels,
                  normalization mean, normalization std, and whether to rescale
                  per image.
        """
        return {
            'size': self.training_images_size,
            'channels': self.training_images_channels,
            'mean': self.training_images_normalise_mean,
            'std': self.training_images_normalise_std,
            'rescale_per_image': self.training_images_rescale_per_image
        }

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook

    def _register_hooks(self):
        for layer_name, layer_idx in self.layers_to_capture.items():
            self.model[layer_idx].register_forward_hook(
                self._hook_fn(layer_name))

    @torch.no_grad()
    def extract_features(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)

        feature_maps = {layer: [] for layer in self.layers_to_capture}
        labels = []

        for batch_imgs, batch_labels in tqdm(dataloader):
            self.activations.clear()
            batch_imgs = batch_imgs.to(self.device)
            _ = self.model(batch_imgs)

            for layer in self.layers_to_capture:
                feature_maps[layer].append(self.activations[layer].cpu())

            labels.append(batch_labels)

        # Concatenate across batches
        feature_maps = {
            layer: torch.cat(feature_maps[layer], dim=0)
            for layer in feature_maps
        }

        return feature_maps, labels

    @torch.no_grad()
    def extract_features_with_pooling(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)

        feature_maps = {layer: [] for layer in self.layers_to_capture}
        labels = []

        for batch_imgs, batch_labels in tqdm(dataloader):
            self.activations.clear()
            batch_imgs = batch_imgs.to(self.device)
            _ = self.model(batch_imgs)

            for layer in self.layers_to_capture:
                raw_feats = self.activations[layer]  # shape: [N, C, H, W]

                if self.pooling_mode == PoolingMode.AVGPOOL:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(
                        raw_feats, (1, 1))  # [N, C, 1, 1]
                    flattened = pooled.view(pooled.size(0), -1)  # [N, C]
                elif self.pooling_mode == PoolingMode.AVGPOOL7X7:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(
                        raw_feats, (7, 7))  # [N, C, 7, 7]
                    flattened = pooled.view(pooled.size(0), -1)  # [N, C×49]
                else:  # PoolingMode.FLATTEN
                    flattened = raw_feats.view(
                        raw_feats.size(0), -1)  # [N, C×H×W]

                feature_maps[layer].append(flattened.cpu())

            labels.append(batch_labels)

        # Concatenate across batches
        feature_maps = {
            layer: torch.cat(feature_maps[layer], dim=0)
            for layer in feature_maps
        }
        labels = torch.cat(labels, dim=0)

        return feature_maps, labels
