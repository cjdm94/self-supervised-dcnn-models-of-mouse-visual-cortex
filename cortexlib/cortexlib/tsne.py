import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from cortexlib.utils.random import GLOBAL_SEED


# TODO: it would be better to have a method compute feats and scores for a single layer - then the caller can loop over layers
class TSNEVisualizer:
    @staticmethod
    def compute_tsne(images_feats, n_pca_components=100, perplexity=30):
        tsne_feats_all_layers = {}

        for layer, feats in images_feats.items():
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)

            feats_pca = PCA(n_components=n_pca_components,
                            random_state=GLOBAL_SEED).fit_transform(feats)
            tsne_feats = TSNE(n_components=2, perplexity=perplexity,
                              random_state=GLOBAL_SEED).fit_transform(feats_pca)
            tsne_feats_all_layers[layer] = tsne_feats

        return tsne_feats_all_layers

    @staticmethod
    def compute_silhouette_scores(tsne_feats_all_layers, class_labels):
        silhouette_scores = {}

        for layer, feats in tsne_feats_all_layers.items():
            score = silhouette_score(feats, class_labels)
            silhouette_scores[layer] = score

        return silhouette_scores

    @staticmethod
    def plot_clusters_all_layers(tsne_feats_all_layers, class_labels, custom_legend_order=None, n_cols=3, palette='tab10'):
        layers = list(tsne_feats_all_layers.keys())
        n_rows = math.ceil(len(layers) / n_cols)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axs = axs.flatten()

        legend_handles, legend_labels = None, None

        for idx, layer in enumerate(layers):
            tsne_feats = tsne_feats_all_layers[layer]
            ax = axs[idx]
            sns.scatterplot(
                x=tsne_feats[:, 0],
                y=tsne_feats[:, 1],
                hue=class_labels,
                palette=palette,
                s=30,
                ax=ax,
                legend=(legend_handles is None)
            )
            ax.set_title(layer)
            ax.set_xticks([])
            ax.set_yticks([])

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

        for ax in axs[len(layers):]:
            ax.axis('off')

        # Legend ordering
        label_order = custom_legend_order or sorted(set(class_labels))
        label_to_handle = dict(zip(legend_labels, legend_handles))
        ordered_handles = [label_to_handle[cls] for cls in label_order]

        fig.legend(ordered_handles, label_order, title='Class',
                   loc='center left', bbox_to_anchor=(0.93, 0.5))

        plt.tight_layout(rect=[0, 0, 0.91, 0.95])
        plt.suptitle("t-SNE on Model Layers", fontsize=16)
        plt.show()

    @staticmethod
    def plot_cluster_single_layer(tsne_feats, class_labels, layer_name="Layer", palette='tab10'):
        plt.figure(figsize=(7, 5))
        ax = sns.scatterplot(
            x=tsne_feats[:, 0],
            y=tsne_feats[:, 1],
            hue=class_labels,
            palette=palette,
            s=30
        )
        ax.set_title(f"t-SNE for {layer_name}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Move legend outside plot
        ax.legend(title='Class', bbox_to_anchor=(1.05, 1),
                  loc='upper left', borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
        plt.show()
