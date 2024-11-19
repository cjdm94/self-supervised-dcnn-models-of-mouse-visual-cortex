# PCA on SimCLR vs. the raw STL10 images

![pca_cum_explained_var_final_layer](/notebooks/002_pca/img/pca_cum_explained_var_final_layer.png)

![pca_cum_explained_var_multi_layers](/notebooks/002_pca/img/pca_explained_var_final_layer.png)

| Number of Principal Components | Raw STL10 Images (% Variance Explained) | SimCLR Representations (% Variance Explained) |
|--------------------------------|------------------------------------------|-----------------------------------------------|
| 12-13                          | **~60%**                                    | ~48%                                          |
| 40                             | **~72%**                    | **~72%**                          |
| 125                            | ~81%                                    | **~90%**                                          |
| 225                            | 86%                                       | **95%**                                            |
| 940                            | 95%                                     | -


### Observation 1: initial principal components of raw image data explain more variance compared to SimCLR representations


### Observation 2: to explain a high level of variance, SimCLR requires fewer principal components compared to the raw image data

| **Category**                  | **Characteristics**                                                                                       | **Variance Explained**             | **Usefulness**                           |
|--------------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------|------------------------------------------|
| **Early Components (Raw Data)** | - Capture **major variance**, dominated by **pixel correlations** (e.g., brightness, gradients).         | High (~majority of variance).       | Limited for downstream tasks, as features are low-level and redundant.        |
| **Later Components (Raw Data)** | - Represent **noise** or small, redundant pixel-level variations with little meaningful information.      | Low (~mostly noise).                | Almost useless, as they lack semantic relevance or discriminative power.      |
| **Early Components (SimCLR)**  | - Capture **broad semantic features** like edges, simple shapes, and overall texture.                     | High (~important broad features).   | Useful for downstream tasks due to decorrelated, informative representations. |
| **Later Components (SimCLR)**  | - Encode **fine-grained, complex semantic features** (e.g., object details, nuanced patterns).            | Moderate (~rich abstract features). | Very useful for downstream tasks, capturing discriminative, high-level details. |


# Comparing SimCLR layers

![pca_cum_explained_var_multi_layers](/notebooks/002_pca/img/pca_cum_explained_var_multi_layers.png)

| Representation        | Number of Features | Components required to explain 95% variance |
|---------------------|--------------------|-|
| STL10      |  27,648            | 940 |
| SimCLR Layer 1    | 36,864            | 3,072 |
| SimCLR Layer 2    | 18,432            | 2,789 |
| SimCLR Layer 4    | 4,608             | 1,176 |
| SimCLR Final Layer | 512               | 225 |

# Cross validation

|   | **Raw STL10 Images**                                                                      | **SimCLR Representations**                                                                  |
|-------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
|            | ![PCA STL10 Cross-Validation](/notebooks/002_pca/img/pca_stl10_cross_validation.png)      | ![PCA SimCLR Cross-Validation](/notebooks/002_pca/img/pca_simclr_cross_validation.png)      |
|       | The first few principal components generalize well because they capture global patterns like brightness gradients or edges, which are consistent across training and test sets. However, later components (higher principal components) seem to overfit, probably to noise. This leads to poor generalization. | SimCLR features are designed to encode meaningful, high-level semantic information. These representations generalize better than raw pixels, and explained variance aligns more closely between training and test sets. |

