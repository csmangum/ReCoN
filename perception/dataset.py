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


def make_house_scene(size=64, noise=0.05, scale_factor=1.0, position_offset=(0, 0)):
    """
    Generate a synthetic house scene with body, roof, and door.

    Creates a simple house composed of:
    - Rectangular body (walls)
    - Triangular roof
    - Rectangular door
    - Optional Gaussian noise for realism

    Args:
        size: Size of the square canvas (default: 64)
        noise: Standard deviation of Gaussian noise to add (default: 0.05)
        scale_factor: Scale factor for the house size (default: 1.0)
        position_offset: (x, y) offset for house position (default: (0, 0))

    Returns:
        numpy.ndarray: Generated house scene as float32 array with values in [0, 1]
    """
    img = canvas(size)

    # Draw house body (rectangular walls)
    # Size: 1/3 of canvas dimensions, scaled and positioned with offsets
    base_width, base_height = int(size//3 * scale_factor), int(size//3 * scale_factor)
    bw, bh = max(6, base_width), max(6, base_height)  # minimum size constraints
    bx = max(0, min(size - bw, size//2 - bw//2 + position_offset[0]))
    by = max(0, min(size - bh, size//2 + position_offset[1]))
    draw_rect(img, bx, by, bw, bh, 0.7)

    # Draw triangular roof above the body
    # Roof spans same width as body, half the height, positioned just above body
    rx, ry = bx, max(0, by - bh//2)  # align with body left edge, position above body
    roof_height = max(3, bh//2)  # minimum roof height
    if ry >= 0 and rx + bw <= size:  # only draw if roof fits in canvas
        draw_triangle(img, rx, ry, bw, roof_height, 1.0)

    # Draw door on the front of the house body
    # Door is small (1/5 body width), tall (half body height), centered on body
    dw, dh = max(2, bw//5), max(3, bh//2)  # door dimensions relative to body
    dx = max(bx, min(bx + bw - dw, bx + bw//2 - dw//2))  # center on body
    dy = max(by, by + bh - dh)  # align with bottom
    if dx + dw <= size and dy + dh <= size:  # only draw if door fits
        draw_rect(img, dx, dy, dw, dh, 0.9)

    # Add Gaussian noise for realism and to make recognition more challenging
    if noise > 0:
        img += noise * np.random.randn(*img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)  # ensure values stay in valid range [0, 1]

    return img


def make_barn_scene(size=64, noise=0.05, scale_factor=1.0, position_offset=(0, 0)):
    """
    Generate a synthetic barn scene with body and arched roof.

    Creates a simple barn composed of:
    - Rectangular body (walls)
    - Arched/curved roof (approximated with multiple rectangles)
    - Large door opening
    - Optional Gaussian noise for realism

    Args:
        size: Size of the square canvas (default: 64)
        noise: Standard deviation of Gaussian noise to add (default: 0.05)
        scale_factor: Scale factor for the barn size (default: 1.0)
        position_offset: (x, y) offset for barn position (default: (0, 0))

    Returns:
        numpy.ndarray: Generated barn scene as float32 array with values in [0, 1]
    """
    img = canvas(size)

    # Draw barn body (wider than house)
    base_width, base_height = int(size//2 * scale_factor), int(size//3 * scale_factor)
    bw, bh = max(8, base_width), max(6, base_height)
    bx = max(0, min(size - bw, size//2 - bw//2 + position_offset[0]))
    by = max(0, min(size - bh, size//2 + position_offset[1]))
    draw_rect(img, bx, by, bw, bh, 0.6)

    # Draw arched roof (approximated with multiple horizontal rectangles)
    roof_height = max(4, bh//2)
    ry = max(0, by - roof_height)
    if ry >= 0:
        # Create arch effect with decreasing width rectangles
        for i in range(roof_height):
            # Width decreases towards top to create arch appearance
            arch_width = bw - (i * 2)
            if arch_width > 4:  # minimum width
                arch_x = bx + (bw - arch_width) // 2
                if arch_x >= 0 and arch_x + arch_width <= size:
                    draw_rect(img, arch_x, ry + i, arch_width, 1, 0.8)

    # Draw large barn door opening (bigger than house door)
    dw, dh = max(4, bw//3), max(4, int(bh * 0.7))
    dx = max(bx, min(bx + bw - dw, bx + bw//2 - dw//2))
    dy = max(by, by + bh - dh)
    if dx + dw <= size and dy + dh <= size:
        draw_rect(img, dx, dy, dw, dh, 0.3)  # darker opening

    # Add Gaussian noise
    if noise > 0:
        img += noise * np.random.randn(*img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

    return img


def make_occluded_scene(size=64, noise=0.05, occlusion_type='tree'):
    """
    Generate a scene with partial occlusion of the main object.

    Args:
        size: Size of the square canvas (default: 64)
        noise: Standard deviation of Gaussian noise to add (default: 0.05)
        occlusion_type: Type of occlusion ('tree', 'cloud', 'box') (default: 'tree')

    Returns:
        numpy.ndarray: Generated occluded scene as float32 array with values in [0, 1]
    """
    # Start with a house scene
    img = make_house_scene(size, noise=0, scale_factor=0.8)

    # Add occlusion based on type
    if occlusion_type == 'tree':
        # Add a simple tree (vertical line + circle approximation)
        tree_x = size // 4
        tree_height = size // 2
        tree_y = size - tree_height
        # Tree trunk
        draw_rect(img, tree_x - 1, tree_y, 3, tree_height, 0.4)
        # Tree foliage (multiple overlapping rectangles to approximate circle)
        foliage_size = size // 6
        for i in range(-foliage_size//2, foliage_size//2, 2):
            for j in range(-foliage_size//2, foliage_size//2, 2):
                if i*i + j*j < (foliage_size//2)**2:  # rough circle check
                    fx, fy = tree_x + i, tree_y + j
                    if 0 <= fx < size-1 and 0 <= fy < size-1:
                        draw_rect(img, fx, fy, 2, 2, 0.5)

    elif occlusion_type == 'cloud':
        # Add cloud-like occlusion (multiple overlapping ovals)
        cloud_y = size // 6
        for i in range(3):
            cloud_x = size//4 + i * 8
            cloud_w, cloud_h = 12, 6
            if cloud_x + cloud_w <= size and cloud_y + cloud_h <= size:
                draw_rect(img, cloud_x, cloud_y, cloud_w, cloud_h, 0.9)

    elif occlusion_type == 'box':
        # Add rectangular obstruction
        box_size = size // 4
        box_x = 3 * size // 4 - box_size
        box_y = size // 2
        if box_x >= 0 and box_y >= 0:
            draw_rect(img, box_x, box_y, box_size, box_size, 0.3)

    # Add noise after occlusion
    if noise > 0:
        img += noise * np.random.randn(*img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

    return img


def make_varied_scene(scene_type='house', size=64, noise=0.05, 
                     scale_range=(0.7, 1.3), position_variance=0.2):
    """
    Generate a scene with random variations in size and position.

    Args:
        scene_type: Type of scene ('house', 'barn', 'occluded') (default: 'house')
        size: Size of the square canvas (default: 64)
        noise: Standard deviation of Gaussian noise to add (default: 0.05)
        scale_range: (min, max) range for random scaling (default: (0.7, 1.3))
        position_variance: Maximum position offset as fraction of size (default: 0.2)

    Returns:
        numpy.ndarray: Generated varied scene as float32 array with values in [0, 1]
    """
    # Random scale factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    
    # Random position offset
    max_offset = int(size * position_variance)
    offset_x = np.random.randint(-max_offset, max_offset + 1)
    offset_y = np.random.randint(-max_offset, max_offset + 1)
    
    if scene_type == 'house':
        return make_house_scene(size, noise, scale_factor, (offset_x, offset_y))
    elif scene_type == 'barn':
        return make_barn_scene(size, noise, scale_factor, (offset_x, offset_y))
    elif scene_type == 'occluded':
        occlusion_types = ['tree', 'cloud', 'box']
        occlusion = np.random.choice(occlusion_types)
        return make_occluded_scene(size, noise, occlusion)
    else:
        # Default to house
        return make_house_scene(size, noise, scale_factor, (offset_x, offset_y))
