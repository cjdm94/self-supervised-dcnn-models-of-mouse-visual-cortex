import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

gabor_filter_colours = {
    'gabor_small': '#cccccc',
    'gabor_medium': '#969696',
    'gabor_large': '#636363',
}

simclr_colours = {
    'layer1': '#f1eef6',
    'layer2': '#bdc9e1',
    'layer3': '#74a9cf',
    'layer4': '#2b8cbe',
    'fc': '#045a8d',
}

vgg19_colours = {
    'conv2_2': '#fef0d9',
    'conv3_4': '#fdcc8a',
    'conv4_4': '#fc8d59',
    'conv5_4': '#e34a33',
    'fc2': '#b30000',
}

neural_colour = "#068235"


def plot_mean_fev(avg_metrics, individual_metrics, remove_gabor=False):
    """
    Plot mean FEV with error bars.
    Parameters:
    - avg_metrics: DataFrame with metrics for each layer of each model, averaged across mice
    - individual_metrics: DataFrame with individual mouse metrics for each layer of each model, with 
    """
    # Clean labels and sort
    ordered_labels = [
        'small', 'medium', 'large',
        'layer1', 'layer2', 'layer3', 'layer4', 'fc',
        'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc2'
    ]
    for df in [avg_metrics, individual_metrics]:
        df["layer"] = df["layer"].str.replace(r"^gabor_", "", regex=True)
        df["layer"] = pd.Categorical(
            df["layer"], categories=ordered_labels, ordered=True)

    avg_metrics = avg_metrics.sort_values('layer')

    # Colour mapping by model
    model_colour = {
        'simclr': simclr_colours['layer4'],
        'vgg19': vgg19_colours['conv5_4'],
        'gabor': gabor_filter_colours['gabor_large']
    }

    avg_metrics["colour"] = avg_metrics["model_target"].str.lower().map(
        lambda s: next((v for k, v in model_colour.items() if k in s), None)
    )

    if remove_gabor:
        # Remove Gabor from plot
        avg_metrics = avg_metrics[~avg_metrics["model_target"].str.contains(
            "gabor", case=False)]
        individual_metrics = individual_metrics[~individual_metrics["model_target"].str.contains(
            "gabor", case=False)]

    # Plot
    plt.figure(figsize=(6, 5))

    # Error bars
    plt.bar(
        avg_metrics['layer'],
        avg_metrics['mean_fev'],
        yerr=avg_metrics['sem_fev'],
        capsize=5,
        color=avg_metrics['colour']
    )

    # Individual dots
    for _, row in individual_metrics.iterrows():
        plt.scatter(
            row["layer"], row["mean_fev"],
            color=model_colour[row["model_target"].split("_")[0]],
            edgecolor='black', s=30, alpha=0.8, zorder=3
        )

    # Axis and legend
    plt.xlabel("Model layer", labelpad=10)
    plt.ylabel("Mean FEV", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.legend(
        handles=[Patch(facecolor=color, label=model.capitalize())
                 for model, color in model_colour.items()],
        title="Model"
    )
    plt.tight_layout()
    return plt
