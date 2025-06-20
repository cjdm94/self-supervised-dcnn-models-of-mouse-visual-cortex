import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GaborFeatureExtractor:
    def __init__(self, ksize=31, thetas=None, gammas=None):
        self.ksize = ksize
        self.thetas = thetas or np.linspace(0, np.pi, 8, endpoint=False)
        self.gammas = gammas or [0.5, 1.0]
        self.size_configs = {
            "gabor_small": {'sigmas': [2.0], 'lambdas': [5.0, 10.0]},
            "gabor_medium": {'sigmas': [4.0], 'lambdas': [10.0]},
            "gabor_large": {'sigmas': [4.0], 'lambdas': [20.0]},
        }

    def extract_features(self, images_tensor):
        with torch.no_grad():
            results = {}
            for name, cfg in self.size_configs.items():
                filters, _ = self._create_filters(
                    cfg['sigmas'], cfg['lambdas'])
                feats = self._apply_filters(images_tensor, filters)
                results[name] = feats
            return results

    def _create_filters(self, sigmas, lambdas):
        filters = []
        labels = []

        for theta in self.thetas:
            for sigma in sigmas:
                for lambd in lambdas:
                    for gamma in self.gammas:
                        kern = cv2.getGaborKernel(
                            (self.ksize, self.ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
                        )
                        kern /= np.sqrt((kern ** 2).sum())
                        filters.append(kern)
                        label = f"θ={round(theta, 2)},σ={sigma},λ={lambd},γ={gamma}"
                        labels.append(label)

        return filters, labels

    def _apply_filters(self, images, filters, batch_size=256):
        filters_tensor = torch.stack(
            [torch.tensor(f, dtype=torch.float32) for f in filters])
        filters_tensor = filters_tensor.unsqueeze(1).to(images.device)

        all_responses = []
        padding = filters_tensor.shape[-1] // 2

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            responses = F.conv2d(batch, filters_tensor, padding=padding)
            pooled = responses.mean(dim=[2, 3])  # mean pooling
            all_responses.append(pooled.cpu())

        return torch.cat(all_responses, dim=0)
