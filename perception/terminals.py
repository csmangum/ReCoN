"""
Terminal feature detectors for ReCoN perception.

This module implements simple feature extraction algorithms that serve as terminal
units in the ReCoN network. These detectors extract basic visual features from
images that can be used to recognize simple geometric shapes and patterns.
"""

import numpy as np
from .dataset import make_house_scene
from scipy import ndimage
import matplotlib.pyplot as plt


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


def sift_like_features(img):
    """
    Extract SIFT-like keypoint features using simple gradient-based detection.
    
    This function approximates SIFT features using:
    - Harris corner detection for keypoints
    - Gradient orientation histograms
    - Local intensity patterns
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary with SIFT-like feature values
    """
    # Compute gradients
    dy, dx = np.gradient(img.astype(np.float32))
    
    # Harris corner detection (simplified)
    # H = [[Ixx, Ixy], [Ixy, Iyy]]
    Ixx = dx * dx
    Ixy = dx * dy  
    Iyy = dy * dy
    
    # Apply Gaussian smoothing
    sigma = 1.0
    Ixx = ndimage.gaussian_filter(Ixx, sigma)
    Ixy = ndimage.gaussian_filter(Ixy, sigma)
    Iyy = ndimage.gaussian_filter(Iyy, sigma)
    
    # Harris response
    det_H = Ixx * Iyy - Ixy * Ixy
    trace_H = Ixx + Iyy
    k = 0.04
    harris_response = det_H - k * (trace_H ** 2)
    
    # Gradient magnitudes and orientations
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    grad_orientation = np.arctan2(dy, dx)
    
    return {
        'corners': float(np.mean(harris_response > 0.001)),  # Corner strength
        'grad_mag': float(np.mean(grad_magnitude)),          # Edge strength
        'grad_std': float(np.std(grad_orientation)),         # Orientation diversity
    }


def blob_detectors(img):
    """
    Detect blob-like structures using Laplacian of Gaussian approximation.
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary with blob detection features
    """
    # Approximated Laplacian of Gaussian using difference of Gaussians
    sigma1, sigma2 = 1.0, 2.0
    blur1 = ndimage.gaussian_filter(img, sigma1)
    blur2 = ndimage.gaussian_filter(img, sigma2)
    dog = blur1 - blur2
    
    # Local maxima detection (simplified)
    local_maxima = ndimage.maximum_filter(np.abs(dog), size=3) == np.abs(dog)
    blob_response = np.abs(dog) * local_maxima
    
    # Pattern analysis
    mean_intensity = ndimage.uniform_filter(img, size=5)
    intensity_variance = ndimage.uniform_filter(img**2, size=5) - mean_intensity**2
    
    return {
        'blobs': float(np.mean(blob_response > 0.01)),        # Blob density
        'texture': float(np.mean(intensity_variance)),        # Local texture
        'contrast': float(np.std(img)),                       # Global contrast
    }


def geometric_features(img):
    """
    Extract geometric features useful for shape recognition.
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary with geometric feature values
    """
    # Thresholded binary image for shape analysis
    threshold = np.mean(img) + np.std(img)
    binary = img > threshold
    
    # Connected components analysis
    labeled_img, num_features = ndimage.label(binary)
    
    # Centroid and moments
    if num_features > 0:
        # Find largest component
        component_sizes = ndimage.sum(binary, labeled_img, range(num_features + 1))
        largest_component = np.argmax(component_sizes[1:]) + 1
        largest_mask = labeled_img == largest_component
        
        # Shape properties
        center_of_mass = ndimage.center_of_mass(largest_mask)
        compactness = np.sum(largest_mask) / (np.sum(largest_mask) ** 0.5) if np.sum(largest_mask) > 0 else 0
        
        # Vertical and horizontal extent
        y_coords, x_coords = np.where(largest_mask)
        if len(y_coords) > 0:
            extent_y = np.max(y_coords) - np.min(y_coords)
            extent_x = np.max(x_coords) - np.min(x_coords)
            aspect_ratio = extent_x / extent_y if extent_y > 0 else 1.0
        else:
            aspect_ratio = 1.0
    else:
        compactness = 0.0
        aspect_ratio = 1.0
    
    return {
        'n_components': float(num_features),
        'compactness': float(compactness * 0.01),  # Scale down
        'aspect_ratio': float(min(aspect_ratio, 3.0) * 0.1),  # Clamp and scale
    }


def advanced_terminals_from_image(img):
    """
    Extract comprehensive terminal features from an image using multiple detectors.
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary mapping terminal IDs to activation values
    """
    # Basic features
    basic = simple_filters(img)
    
    # Advanced features
    sift_feats = sift_like_features(img)
    blob_feats = blob_detectors(img)
    geom_feats = geometric_features(img)
    
    # Combine all features with appropriate scaling
    terminals = {
        # Basic terminals (existing)
        't_mean': basic['mean'],
        't_vert': basic['vert'] * 0.1,
        't_horz': basic['horz'] * 0.1,
        
        # SIFT-like terminals
        't_corners': sift_feats['corners'],
        't_edges': sift_feats['grad_mag'] * 0.5,
        't_orient_var': sift_feats['grad_std'] * 0.2,
        
        # Blob detection terminals
        't_blobs': blob_feats['blobs'],
        't_texture': blob_feats['texture'] * 0.5,
        't_contrast': blob_feats['contrast'] * 0.3,
        
        # Geometric terminals
        't_n_shapes': geom_feats['n_components'] * 0.1,
        't_compact': geom_feats['compactness'],
        't_aspect': geom_feats['aspect_ratio'],
    }
    
    return terminals


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


def advanced_sample_scene_and_terminals():
    """
    Generate a random synthetic scene and extract advanced terminal features.

    Returns:
        tuple: (image, terminals) where:
               - image: Generated 2D numpy array representing the scene
               - terminals: Dictionary of comprehensive terminal unit activations
    """
    img = make_house_scene()
    t = advanced_terminals_from_image(img)
    return img, t
