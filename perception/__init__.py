"""
Perception Package.

This package provides utilities for synthetic scene generation and feature extraction
that serve as the sensory/perceptual layer for ReCoN networks. It includes:

- Synthetic dataset generation with geometric shapes
- Basic feature extraction algorithms (edge detection, intensity)
- Terminal unit activation mapping from visual features
- Optional denoising autoencoder features (training gated via env `RECON_TRAIN_AE`)

These components enable ReCoN networks to process visual information and learn
to recognize patterns in 2D scenes through hierarchical feature composition. Heavy
visualization imports are lazy/optional to keep the core lightweight.
"""

# Perception Package
