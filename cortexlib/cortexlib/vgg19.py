import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


class PreTrainedVGG19Model:
    def __init__(self, layers_to_capture=None, batch_size=16, num_workers=4, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.vgg19(
            pretrained=True).features.to(self.device).eval()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Default to conv3_1 if nothing passed
        self.layers_to_capture = layers_to_capture or {
            "conv3_1": 10
        }
        self.activations = {}
        self._register_hooks()

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
        # labels = []

        for batch_imgs, _ in tqdm(dataloader):
            self.activations.clear()
            batch_imgs = batch_imgs.to(self.device)
            _ = self.model(batch_imgs)

            for layer in self.layers_to_capture:
                feature_maps[layer].append(self.activations[layer].cpu())

            # labels.append(batch_labels)

        # Concatenate across batches
        feature_maps = {
            layer: torch.cat(feature_maps[layer], dim=0)
            for layer in feature_maps
        }

        return feature_maps
