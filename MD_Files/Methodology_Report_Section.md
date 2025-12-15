# Methodology

## 1. Architectural Design
Our proposed model integrates a Convolutional Neural Network (CNN) backbone with a Transformer encoder to leverage both local feature extraction and global context modeling. The architecture consists of two main stages: a feature extraction stage using a ResNet-18 backbone and a global reasoning stage using a Transformer Encoder.

**CNN Backbone (ResNet-18):**
We utilize a ResNet-18 model pre-trained on ImageNet as the primary feature extractor. We remove the original fully connected layer and the final average pooling layer. The input images, resized to $64 \times 64$, are processed by the ResNet layers to produce a feature map of shape $(512, 2, 2)$. This stage is responsible for capturing low-level visual patterns such as edges, textures, and object parts.

**Fusion Adapter:**
To bridge the CNN output with the Transformer, we apply a $1 \times 1$ convolution layer to project the channel dimension from $512$ down to $256$, designated as the embedding dimension ($d_{model}$). The resulting $(256, 2, 2)$ tensor is then flattened along the spatial dimensions and permuted to form a sequence of length $4$ (i.e., shape $(B, 4, 256)$). This transformation effectively converts the spatial grid into a sequence of feature patches suitable for attention mechanisms.

**Transformer Encoder:**
The core fusion mechanism is a standard Transformer Encoder consisting of 2 layers. Each layer employs Multi-Head Self-Attention with 4 heads and a feed-forward network. This allows each of the 4 spatial patches to attend to every other patch, modeling long-range dependencies across the image that the local receptive fields of the CNN might miss.

**Classification Head:**
The output sequence from the Transformer is aggregated via global average pooling to produce a single context vector of size $256$. This vector is passed through a final linear layer to output the logits for the 10 CIFAR-10 classes.

## 2. Unification Strategy (What & Why)
We fused **ResNet-18** and a **Transformer** to address the limitations of using either architecture in isolation.
*   **What was fused:** A pretrained residual network for spatial feature extraction and a self-attention encoder for sequence modeling.
*   **Why:** CNNs are efficient at processing pixel-level redundancy but struggle with global context in lower layers. Transformers excel at global reasoning but are computationally expensive on raw pixels. By cascading them, we use the CNN to "abbreviate" the high-resolution image into rich feature tokens, which the Transformer then contextualizes.

## 3. Implementation Details

**Data Preprocessing:**
The CIFAR-10 dataset ($32 \times 32$ RGB images) is used. During training, we apply the following augmentations:
1.  **Upscaling**: Images are resized to $64 \times 64$ to provide sufficient spatial resolution for the ResNet backbone.
2.  **Augmentation**: Random horizontal flips are applied.
3.  **Normalization**: Pixel values are normalized to a range of $[-1, 1]$ using mean $(0.5, 0.5, 0.5)$ and standard deviation $(0.5, 0.5, 0.5)$.

**Training Configuration:**
*   **Framework**: PyTorch & PyTorch Lightning.
*   **Optimizer**: AdamW with a learning rate of $1 \times 10^{-4}$.
*   **Loss Function**: Cross-Entropy Loss.
*   **Batch Size**: 64.
*   **Epochs**: 10.
*   **Hardware**: Training is accelerated via CUDA (GPU) if available.
