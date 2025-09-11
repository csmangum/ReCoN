"""
Terminal feature detectors for ReCoN perception.

This module implements simple feature extraction algorithms that serve as terminal
units in the ReCoN network. These detectors extract basic visual features from
images that can be used to recognize simple geometric shapes and patterns.
"""

import numpy as np
from .dataset import make_house_scene


def simple_filters(img):
    """
    Extract basic visual features using simple convolution filters.

    This function applies basic edge detection filters to extract:
    - Mean intensity: Overall brightness of the image
    - Vertical edges: Strength of vertical edge responses
    - Horizontal edges: Strength of horizontal edge responses

    The edge detection uses a simple 2x2 kernel that responds to intensity
    differences between adjacent pixels.

    Args:
        img: Input image as 2D numpy array

    Returns:
        dict: Dictionary with keys 'mean', 'vert', 'horz' containing
              the extracted feature values as floats
    """
    # mean intensity, vertical edge proxy, horizontal edge proxy
    # Create a 2x2 edge detection kernel: [[1, -1], [1, -1]]
    # This kernel highlights vertical transitions (left-right intensity differences)
    k = np.array([[1,-1],[1,-1]], dtype=np.float32)
    kv = k  # vertical edge kernel (responds to horizontal intensity changes)
    kh = k.T  # horizontal edge kernel (transposed, responds to vertical intensity changes)

    # Apply convolution and take absolute value to measure edge strength
    # mode='valid' ensures we only get valid convolutions (no padding artifacts)
    vert = np.abs(np.convolve(img.flatten(), kv.flatten(), mode='valid')).mean()
    horz = np.abs(np.convolve(img.flatten(), kh.flatten(), mode='valid')).mean()
    mean = img.mean()

    return {'mean': float(mean), 'vert': float(vert), 'horz': float(horz)}


def terminals_from_image(img):
    """
    Convert image features to terminal unit activations.

    This function maps the extracted visual features to activation values
    suitable for ReCoN terminal units. The features are scaled appropriately
    to work with the network's activation dynamics.

    Args:
        img: Input image as 2D numpy array

    Returns:
        dict: Dictionary mapping terminal IDs to activation values:
              - 't_mean': Overall brightness activation
              - 't_vert': Vertical edge detection activation
              - 't_horz': Horizontal edge detection activation
    """
    feats = simple_filters(img)
    # map to three terminal nodes
    return {
        't_mean': feats['mean'],
        't_vert': feats['vert']*0.1,
        't_horz': feats['horz']*0.1,
    }


def sample_scene_and_terminals():
    """
    Generate a random synthetic scene and extract its terminal features.

    This convenience function creates a new synthetic house scene and
    immediately extracts the terminal features from it. Useful for
    testing and demonstration purposes.

    Returns:
        tuple: (image, terminals) where:
               - image: Generated 2D numpy array representing the scene
               - terminals: Dictionary of terminal unit activations
    """
    img = make_house_scene()
    t = terminals_from_image(img)
    return img, t
