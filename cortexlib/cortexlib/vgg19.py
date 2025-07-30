import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from enum import Enum
import warnings

VGG19_LAYER_INDEX = {
    # Features
    "conv1_1": ("features", 0), "conv1_2": ("features", 2),
    "conv2_1": ("features", 5), "conv2_2": ("features", 7),
    "conv3_1": ("features", 10), "conv3_2": ("features", 12),
    "conv3_3": ("features", 14), "conv3_4": ("features", 16),
    "conv4_1": ("features", 19), "conv4_2": ("features", 21),
    "conv4_3": ("features", 23), "conv4_4": ("features", 25),
    "conv5_1": ("features", 28), "conv5_2": ("features", 30),
    "conv5_3": ("features", 32), "conv5_4": ("features", 34),

    # Classifier
    "fc1": ("classifier", 0),
    "fc2": ("classifier", 3),
    "fc3": ("classifier", 6),
}


class PoolingMode(Enum):
    FLATTEN = "flatten"
    AVGPOOL = "avgpool"
    AVGPOOL7X7 = "avgpool7x7"


class PreTrainedVGG19Model:
    def __init__(self, layers_to_capture=None, batch_size=16, num_workers=4, pooling_mode=PoolingMode.FLATTEN, device=None):
        # ImageNet training settings
        self.training_images_size = (224, 224)
        self.training_images_normalise_mean = [0.485, 0.456, 0.406]
        self.training_images_normalise_std = [0.229, 0.224, 0.225]
        self.training_images_channels = 3  # RGB images
        self.training_images_rescale_per_image = False

        self.pooling_mode = pooling_mode
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.model = torchvision.models.vgg19(
                pretrained=True).to(self.device).eval()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # mapping VGG-19 layers to SimCLR by aligning their hierarchical depth: early SimCLR layers (e.g., layer1, layer2)
        # are compared to early VGG-19 conv layers (e.g., conv2_2, conv3_4), and later SimCLR layers (layer3, layer4, fc)
        # to deeper VGG-19 layers (conv4_4, conv5_4, fc). For fair comparison, VGG feature maps are spatially averaged to
        # match the dimensionality of SimCLR outputs
        layer_names = layers_to_capture or [
            "conv2_2", "conv3_4", "conv4_4", "conv5_4", "fc2"]
        self.layers_to_capture = {
            name: VGG19_LAYER_INDEX[name] for name in layer_names
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
        for layer_name, (section, idx) in self.layers_to_capture.items():
            submodule = getattr(self.model, section)
            submodule[idx].register_forward_hook(self._hook_fn(layer_name))

    @torch.no_grad()
    def extract_features_with_pooling(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)

        feature_maps = {layer: [] for layer in self.layers_to_capture}
        labels = []

        for batch_imgs, batch_labels in tqdm(dataloader):
            self.activations.clear()
            batch_imgs = batch_imgs.to(self.device)

            x = self.model.features(batch_imgs)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            _ = self.model.classifier(x)

            for layer in self.layers_to_capture:
                raw_feats = self.activations[layer]

                if raw_feats.ndim == 4:
                    if self.pooling_mode == PoolingMode.AVGPOOL:
                        pooled = torch.nn.functional.adaptive_avg_pool2d(
                            raw_feats, (1, 1))  # [N, C, 1, 1]
                        flattened = pooled.view(pooled.size(0), -1)  # [N, C]
                    elif self.pooling_mode == PoolingMode.AVGPOOL7X7:
                        pooled = torch.nn.functional.adaptive_avg_pool2d(
                            raw_feats, (7, 7))  # [N, C, 7, 7]
                        flattened = pooled.view(
                            pooled.size(0), -1)  # [N, C×49]
                    else:  # FLATTEN
                        flattened = raw_feats.view(raw_feats.size(0), -1)
                else:
                    # For FC layers or other non-4D outputs
                    flattened = raw_feats.view(raw_feats.size(0), -1)

                feature_maps[layer].append(flattened.cpu())

            labels.append(batch_labels)

        # Concatenate across batches
        feature_maps = {
            layer: torch.cat(feature_maps[layer], dim=0)
            for layer in feature_maps
        }
        labels = torch.cat(labels, dim=0)

        return feature_maps, labels

    def _get_layer(self, layer_name):
        section, idx = VGG19_LAYER_INDEX[layer_name]
        return getattr(self.model, section)[idx] if section == "features" else getattr(self.model, section)[idx]

    def capture_single_forward(self, img_tensor, target_layer):
        activations = {}

        def hook_fn(module, input, output):
            activations[target_layer] = output

        layer = self._get_layer(target_layer)
        handle = layer.register_forward_hook(hook_fn)

        x = img_tensor.to(self.device)
        x = x.repeat(1, 3, 1, 1)

        feats = self.model.features(x)
        pooled = self.model.avgpool(feats)
        flat = torch.flatten(pooled, 1)
        _ = self.model.classifier(flat)

        handle.remove()

        out = activations[target_layer]

        # 🔧 Match training: apply avgpool if 4D feature map
        if out.ndim == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)

        else:
            out = out.view(out.size(0), -1)

        return out

    @staticmethod
    def l2_penalty(img, lam=0.0001):
        l2_penalty = lam * torch.sum(img ** 2)
        return l2_penalty

    def generate_synthetic_image(self, layer_name, ridge_model, iterations=200, lr=0.05, l2_lam=1e-3):
        ridge_weights = torch.tensor(
            ridge_model.coef_, dtype=torch.float32, device=self.device).unsqueeze(0)
        synthetic_image = torch.randn(
            1, 1, 224, 224, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([synthetic_image], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()

            # Do NOT repeat here
            feats = self.capture_single_forward(synthetic_image, layer_name)
            feats = feats.view(1, -1)

            score = torch.matmul(feats, ridge_weights.t()).squeeze()
            loss = -score + (self.l2_penalty(synthetic_image,
                             l2_lam) if l2_lam is not None else 0)

            loss.backward()
            optimizer.step()
            synthetic_image.data.clamp_(0, 1)

        img_np = synthetic_image.detach().cpu().squeeze().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        return img_np
