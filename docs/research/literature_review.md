# Literature Review: Spatio-Temporal Graph Neural Networks

## Overview
We reviewed the landscape of Spatio-Temporal Graph Neural Networks (STGNNs) up through 2025/2026 for traffic forecasting. The progression of research highlights the move from static spatial topologies to dynamic and adaptive graph structures.

## Key Models in the Field

1. **STGCN (Spatio-Temporal Graph Convolutional Network)**
   - *Method*: Combines Chebyshev polynomial-based graph convolution with 1D temporal convolutions.
   - *Limitation*: Relies entirely on a static, predefined adjacency matrix based on physical road distances. Does not account for dynamic traffic conditions or hidden dependencies.

2. **DCRNN (Diffusion Convolutional Recurrent Neural Network)**
   - *Method*: Integrates graph convolutions modeled as a diffusion process into a Recurrent Neural Network (GRU).
   - *Limitation*: RNNs suffer from slow training speeds and vanishing gradients over long sequences compared to fully convolutional architectures.

3. **ASTGCN (Attention Based Spatial-Temporal Graph Convolutional Networks)**
   - *Method*: Applies spatial and temporal attention mechanisms before the graph convolutions to dynamically weigh the importance of nodes and time steps.
   - *Limitation*: Can be computationally heavy due to the large attention matrices ($N \times N$ and $T \times T$).

4. **Graph WaveNet**
   - *Method*: Uses dilated 1D convolutions for time and introduces an **Adaptive Adjacency Matrix** alongside the static one. It learns hidden spatial dependencies end-to-end without needing a predefined graph.
   - *Limitation*: Requires tuning of the embedding dimensions for the adaptive graph to prevent overfitting.

## Identification of Recurring Limitations
1. **Over-reliance on Static Graphs**: Traditional models (like STGCN) assume traffic only flows along predefined roads. In reality, distant nodes might exhibit correlated traffic patterns (e.g., business districts that are far apart but share rush hour peaks).
2. **Computational Inefficiency**: RNN-based and heavy self-attention models scale poorly as the number of sensors ($N$) increases.

## Selected Research Direction: Adaptive Graph Convolution
**Option A/F Hybrid**: We propose an "Adaptive Graph STGNN". 
Instead of relying purely on the physical adjacency matrix `adj_METR-LA.pkl`, we will introduce a learned Node Embedding layer that generates an **Adaptive Adjacency Matrix** during training (inspired by Graph WaveNet). This falls under the *Dynamic Graph* methodology and allows the model to learn hidden spatial relationships that the physical map misses, while maintaining the lightweight convolutional nature of STGCN.
