## Feature information


| **Representation**    | **Features per Image (C x H × W)** | **# Channels (C)** | **# Spatial Pixels (H × W)** | **Type**            | **Explanation**                                                                                     |
|------------------------|------------------------|---------------------|-----------------------------|---------------------|-----------------------------------------------------------------------------------------------------|
| **Raw Image Data**     | 150,528               | 3                   | 50,176                      | Input | Represents the raw pixel data of the input image.                                                  |
| **SimCLR Layer 1**     | 200,704               | 64                  | 3,136                       | Convolutional       | Extracts low-level features (e.g., edges and textures) using \(64\) convolutional filters.          |
| **SimCLR Layer 2**     | 100,352               | 128                 | 784                         | Convolutional       | Builds on Layer 1 features, extracting more abstract patterns with \(128\) convolutional filters.   |
| **SimCLR Layer 4**     | 25,088                | 512                 | 49                          | Convolutional       | Extracts highly abstract and semantic features using \(512\) convolutional filters.                |
| **SimCLR Final Layer** | 512                   | 512                 | 1                           | Fully Connected     | A global feature vector produced by global average pooling over the final layer's spatial features. |


