"""
Synthetic dataset generation for ReCoN perception experiments.

This module provides utilities for generating synthetic 2D scenes with simple geometric
shapes that can be used to test and demonstrate the ReCoN perception pipeline. The
generated scenes represent basic objects like houses made of rectangles and triangles.
"""

import numpy as np


def canvas(size=64):
    """
    Create a blank canvas for drawing synthetic scenes.

    Args:
        size: Size of the square canvas (default: 64x64)

    Returns:
        numpy.ndarray: Zero-initialized float32 array of shape (size, size)
    """
    return np.zeros((size, size), dtype=np.float32)


def draw_rect(img, x, y, w, h, val=1.0):
    """
    Draw a filled rectangle on an image canvas.

    Args:
        img: Image array to draw on (modified in-place)
        x: X-coordinate of top-left corner
        y: Y-coordinate of top-left corner
        w: Width of the rectangle
        h: Height of the rectangle
        val: Intensity value to fill the rectangle with (default: 1.0)

    Returns:
        numpy.ndarray: The modified image array (same as input img)
    """
    img[y:y+h, x:x+w] = val
    return img


def draw_triangle(img, x, y, base, height, val=1.0):
    """
    Draw a filled isosceles triangle on an image canvas.

    The triangle is drawn with its base at the bottom and apex at the top.
    Uses a scanline approach where each row's width increases symmetrically
    from the apex downward to create the triangular shape.

    Args:
        img: Image array to draw on (modified in-place)
        x: X-coordinate of the leftmost point of the base
        y: Y-coordinate of the apex (top point)
        base: Width of the triangle base
        height: Height of the triangle
        val: Intensity value to fill the triangle with (default: 1.0)

    Returns:
        numpy.ndarray: The modified image array (same as input img)
    """
    for i in range(height):
        # Calculate the left and right boundaries for this scanline
        # At row i (from top), the triangle width increases symmetrically
        # The formula creates wider rows as we move down from the apex
        start = x + (height - i - 1)  # left boundary moves right as i increases
        end = x + base - (height - i - 1)  # right boundary moves left as i increases
        img[y+i, start:end] = val
    return img


def make_house_scene(size=64, noise=0.05):
    """
    Generate a synthetic house scene with body, roof, and door.

    Creates a simple house composed of:
    - Rectangular body (walls)
    - Triangular roof
    - Rectangular door
    - Optional Gaussian noise for realism

    The house is centered in the canvas with proportions based on the canvas size.

    Args:
        size: Size of the square canvas (default: 64)
        noise: Standard deviation of Gaussian noise to add (default: 0.05)

    Returns:
        numpy.ndarray: Generated house scene as float32 array with values in [0, 1]
    """
    img = canvas(size)

    # Draw house body (rectangular walls)
    # Size: 1/3 of canvas dimensions, centered horizontally, positioned in lower half
    bw, bh = size//3, size//3  # body width and height
    bx, by = size//2 - bw//2, size//2  # center horizontally, position at canvas middle
    draw_rect(img, bx, by, bw, bh, 0.7)

    # Draw triangular roof above the body
    # Roof spans same width as body, half the height, positioned just above body
    rx, ry = bx, by - bh//2  # align with body left edge, position above body
    draw_triangle(img, rx, ry, bw, bh//2, 1.0)

    # Draw door on the front of the house body
    # Door is small (1/5 body width), tall (half body height), centered on body
    dw, dh = bw//5, bh//2  # door dimensions relative to body
    dx, dy = bx + bw//2 - dw//2, by + bh - dh  # center on body, align with bottom
    draw_rect(img, dx, dy, dw, dh, 0.9)

    # Add Gaussian noise for realism and to make recognition more challenging
    img += noise * np.random.randn(*img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)  # ensure values stay in valid range [0, 1]

    return img
