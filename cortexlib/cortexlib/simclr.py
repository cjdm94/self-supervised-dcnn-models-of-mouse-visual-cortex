import os
from urllib.error import HTTPError
import urllib.request
from torch.utils.data import Dataset
from typing import Dict
from tqdm.notebook import tqdm
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path
from cortexlib.file import find_project_root


class PreTrainedSimCLRModel(nn.Module):
    def __init__(self, hidden_dim=128, intermediate_layers=['layer1', 'layer2', 'layer3', 'layer4']):
        super().__init__()

        # Base ResNet18 backbone (pretrained=False, because we load custom weights later, from the SimCLR checkpoint file)
        self.convnet = torchvision.models.resnet18(pretrained=False)

        # This is the projection head, only needed during training. For downstream tasks it is disposed of
        # and the final linear layer output is used (Chen et al., 2020)
        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.num_workers = os.cpu_count()
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.load_pretrained()

        self.intermediate_layer_features = {}
        self.set_intermediate_layers_to_capture(intermediate_layers)

    def load_pretrained(self):
        """
        Load pretrained SimCLR weights
        """
        base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial17/"
        current_file = Path(__file__)
        project_root = find_project_root(current_file)
        models_dir = project_root / "models"
        pretrained_simclr_filename = "SimCLR.ckpt"
        pretrained_simclr_path = models_dir / pretrained_simclr_filename

        models_dir.mkdir(parents=True, exist_ok=True)

        # Check whether the pretrained model file already exists locally. If not, try downloading it
        file_url = base_url + pretrained_simclr_filename
        if not os.path.isfile(pretrained_simclr_path):
            print(f"Downloading pretrained SimCLR model {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, pretrained_simclr_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)

        print(f"Already downloaded pretrained model: {file_url}")

        # Load pretrained model
        checkpoint = torch.load(pretrained_simclr_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.to(self.device)
        self.eval()

    def set_intermediate_layers_to_capture(self, layers):
        """
        Register hooks to capture features from intermediate layers
        """
        # Just check the layers specified are actually in the convnet
        top_level_block_layers = [name for name,
                                  _ in self.convnet.named_children()]
        if not all(layer in top_level_block_layers for layer in layers):
            print('You have specified convnet layers that are not top-level blocks - make sure your layer names are valid')

        self.intermediate_layers_to_capture = layers
        intermediate_layer_features = {}

        def get_hook(layer_name):
            def hook(module, input, output):
                intermediate_layer_features[layer_name] = output.detach()
            return hook

        for layer_name in layers:
            layer = dict([*self.convnet.named_modules()])[layer_name]
            layer.register_forward_hook(get_hook(layer_name))

        self.intermediate_layer_features = intermediate_layer_features

    @torch.no_grad()
    def extract_features(self, dataset: Dataset) -> Dict[str, torch.Tensor]:
        """
        Run the pretrained SimCLR model on the image data, and capture features from final layer and intermediate layers.

        Args:
            dataset (Dataset): A PyTorch Dataset containing input images and labels. The image data should have shape (N, C, H, W)

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - Intermediate layer features as tensors.
                - Final layer features under 'final_layer'.
                - Labels under 'labels'.
            Features from a given layer has shape (N, F) where N is num images, F is number of features - flattened version of (C, H, W).
        """
        self.convnet.fc = nn.Identity()  # Removing projection head g(.)
        self.eval()
        self.to(self.device)

        # Encode all images
        data_loader = DataLoader(
            dataset, batch_size=64, num_workers=self.num_workers, shuffle=False, drop_last=False)
        feats, labels, intermediate_features = [], [], {
            layer: [] for layer in self.intermediate_layers_to_capture}

        for batch_idx, (batch_imgs, batch_labels) in enumerate(tqdm(data_loader)):
            batch_imgs = batch_imgs.to(self.device)
            batch_feats = self.convnet(batch_imgs)

            feats.append(batch_feats.detach().cpu())
            labels.append(batch_labels)

            # Collect intermediate layer outputs
            for layer in self.intermediate_layers_to_capture:
                # Final linear layer outputs a 2d tensor; but intermediate layers don't, so we flatten them (ready for PCA etc.)
                # layer_output_flattened = self.intermediate_layer_features[layer].view(self.intermediate_layer_features[layer].size(0), -1)
                # intermediate_features[layer].append(layer_output_flattened.cpu())

                # DON'T FLATTEN - IT CAUSES PROBLEMS WHEN VISUALISING FEATURES LATER
                intermediate_features[layer].append(
                    self.intermediate_layer_features[layer].cpu())

        # Concatenate results for each layer
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        intermediate_features = {layer: torch.cat(
            intermediate_features[layer], dim=0) for layer in self.intermediate_layers_to_capture}

        # Debugging log after concatenation
        print("✅ Feature extraction complete. Final feature shapes:")
        print(f"Final layer: {feats.shape}")
        for layer, feature in intermediate_features.items():
            print(f"{layer}: {feature.shape}")  # Check final stored shape

        return {**intermediate_features, 'fc': feats}, labels
