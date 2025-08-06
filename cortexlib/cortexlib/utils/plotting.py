import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

gabor_filter_colours = {
    'gabor_small': '#cccccc',
    'gabor_medium': '#969696',
    'gabor_large': '#636363',
}

simclr_colours = {
    'layer1': '#eff3ff',
    'layer2': '#bdd7e7',
    'layer3': '#6baed6',
    'layer4': '#3182bd',
    'fc': '#08519c',
}

vgg19_colours = {
    'conv2_2': '#fee5d9',
    'conv3_4': '#fcae91',
    'conv4_4': '#fb6a4a',
    'conv5_4': '#de2d26',
    'fc2': '#a50f15',
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


def plot_fev_vs_metric_scatter(df, colours, metric_key, metric_title):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FormatStrFormatter

    fig = plt.figure(figsize=(8, 8))  # physical size of full figure
    gs = GridSpec(1, 1, figure=fig, left=0.2, right=0.8,
                  top=0.8, bottom=0.2)  # tightly control margins
    ax = fig.add_subplot(gs[0])

    for _, row in df.iterrows():
        color = colours.get(row["layer"], "black")
        ax.errorbar(
            x=row[metric_key],
            y=row["mean_fev"],
            yerr=row["sem_mean_fev_plot"],
            fmt='o',
            markersize=10,
            capsize=3,
            ecolor='gray',
            color=color,
            markeredgecolor='black',
            markeredgewidth=0.6
        )
        ax.annotate(
            row["layer"],
            xy=(row[metric_key], row["mean_fev"]),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=11,
            va='center',
            ha='left',
        )

    ax.set_xlabel(metric_title, labelpad=10)
    ax.set_ylabel("Mean FEV ± SEM", labelpad=10)
    ax.set_box_aspect(1)  # square plot area
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    return fig
