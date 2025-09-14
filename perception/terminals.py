"""
Terminal feature detectors for ReCoN perception.

This module implements simple feature extraction algorithms that serve as terminal
units in the ReCoN network. These detectors extract basic visual features from
images that can be used to recognize simple geometric shapes and patterns.
"""

import numpy as np
from .dataset import make_house_scene

# Optional SciPy dependency: provide fallbacks if unavailable
try:
    from scipy import ndimage as _sp_ndimage  # type: ignore

    def _gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
        return _sp_ndimage.gaussian_filter(img, sigma)

    def _maximum_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
        return _sp_ndimage.maximum_filter(img, size=size)

    def _uniform_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
        return _sp_ndimage.uniform_filter(img, size=size)

    def _label(binary: np.ndarray):
        return _sp_ndimage.label(binary)

    def _center_of_mass(mask: np.ndarray):
        return _sp_ndimage.center_of_mass(mask)
    
    def _sum_by_label(input_arr: np.ndarray, labels: np.ndarray, index) -> np.ndarray:
        return _sp_ndimage.sum(input_arr, labels, index)
except Exception:  # pragma: no cover - provide lightweight fallbacks
    # Lightweight NumPy-based fallbacks for environments without SciPy
    def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
        radius = max(1, int(3 * float(sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
        s = kernel.sum()
        if s <= 0:
            return np.array([1.0], dtype=np.float32)
        return (kernel / s).astype(np.float32)

    def _apply_convolve1d_same(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
        # Apply 1D convolution along specified axis using 'same' mode
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis, img)

    def _gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return img.astype(np.float32)
        k = _gaussian_kernel_1d(sigma)
        tmp = _apply_convolve1d_same(img.astype(np.float32), k, axis=1)
        out = _apply_convolve1d_same(tmp, k, axis=0)
        return out.astype(np.float32)

    def _maximum_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
        # 2D morphological max over size x size window via shift-and-max
        size = int(max(1, size))
        pad = size // 2
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
        h, w = img.shape
        acc = None
        for dy in range(size):
            for dx in range(size):
                window = padded[dy:dy+h, dx:dx+w]
                if acc is None:
                    acc = window.copy()
                else:
                    acc = np.maximum(acc, window)
        return acc.astype(np.float32)

    def _uniform_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
        # Separable box filter using 1D moving average along rows then columns
        size = int(max(1, size))
        k1d = np.ones(size, dtype=np.float32) / float(size)
        tmp = _apply_convolve1d_same(img.astype(np.float32), k1d, axis=1)
        out = _apply_convolve1d_same(tmp, k1d, axis=0)
        return out.astype(np.float32)

    def _label(binary: np.ndarray):
        # Simple connected components labeling (4-neighborhood)
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or labels[y, x] != 0:
                    continue
                current_label += 1
                # BFS flood fill
                stack = [(y, x)]
                labels[y, x] = current_label
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                stack.append((ny, nx))
        return labels, current_label

    def _center_of_mass(mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return (0.0, 0.0)
        return (float(ys.mean()), float(xs.mean()))
    
    def _sum_by_label(input_arr: np.ndarray, labels: np.ndarray, index) -> np.ndarray:
        # Compute sum of input over each label in index (iterable or int)
        if isinstance(index, int):
            mask = labels == index
            return float(input_arr[mask].sum())
        sums = []
        for idx in index:
            mask = labels == idx
            sums.append(float(input_arr[mask].sum()))
        return np.array(sums, dtype=np.float32)
# Lazy import matplotlib only if needed (debug/plotting)
try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:  # pragma: no cover - optional
    plt = None
import pickle
import os


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
    Ixx = _gaussian_filter(Ixx, sigma)
    Ixy = _gaussian_filter(Ixy, sigma)
    Iyy = _gaussian_filter(Iyy, sigma)
    
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
    blur1 = _gaussian_filter(img, sigma1)
    blur2 = _gaussian_filter(img, sigma2)
    dog = blur1 - blur2
    
    # Local maxima detection (simplified)
    local_maxima = _maximum_filter(np.abs(dog), size=3) == np.abs(dog)
    blob_response = np.abs(dog) * local_maxima
    
    # Pattern analysis
    mean_intensity = _uniform_filter(img, size=5)
    intensity_variance = _uniform_filter(img**2, size=5) - mean_intensity**2
    
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
    labeled_img, num_features = _label(binary)
    
    # Centroid and moments
    if num_features > 0:
        # Find largest component
        component_sizes = _sum_by_label(binary, labeled_img, range(num_features + 1))
        largest_component = np.argmax(component_sizes[1:]) + 1
        largest_mask = labeled_img == largest_component
        
        # Shape properties
        center_of_mass = _center_of_mass(largest_mask)
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


class SimpleAutoencoder:
    """
    A simple denoising autoencoder for extracting compressed patch features.
    
    This lightweight autoencoder learns to reconstruct small image patches
    and provides the compressed representation as terminal features for ReCoN.
    """
    
    def __init__(self, patch_size=8, latent_dim=4, noise_factor=0.1):
        """
        Initialize the autoencoder.
        
        Args:
            patch_size: Size of square patches to extract (default: 8x8)
            latent_dim: Dimension of compressed representation (default: 4)
            noise_factor: Amount of noise to add for denoising training (default: 0.1)
        """
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        
        # Network dimensions
        self.input_dim = patch_size * patch_size
        self.hidden_dim = max(8, latent_dim * 2)
        
        # Initialize weights with small random values
        self.W_encoder = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b_encoder = np.zeros(self.hidden_dim)
        self.W_latent = np.random.randn(self.hidden_dim, self.latent_dim) * 0.1
        self.b_latent = np.zeros(self.latent_dim)
        
        self.W_decode = np.random.randn(self.latent_dim, self.hidden_dim) * 0.1
        self.b_decode = np.zeros(self.hidden_dim)
        self.W_output = np.random.randn(self.hidden_dim, self.input_dim) * 0.1
        self.b_output = np.zeros(self.input_dim)
        
        self.is_trained = False
    
    def _sigmoid(self, x):
        """Sigmoid activation function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _extract_patches(self, img, n_patches=20):
        """Extract random patches from an image."""
        patches = []
        h, w = img.shape
        
        for _ in range(n_patches):
            # Random top-left corner ensuring patch fits in image
            y = np.random.randint(0, max(1, h - self.patch_size))
            x = np.random.randint(0, max(1, w - self.patch_size))
            
            patch = img[y:y+self.patch_size, x:x+self.patch_size]
            
            # Handle edge cases where patch might be smaller than expected
            if patch.shape != (self.patch_size, self.patch_size):
                patch = np.pad(patch, 
                             ((0, self.patch_size - patch.shape[0]),
                              (0, self.patch_size - patch.shape[1])),
                             mode='constant', constant_values=0)
            
            patches.append(patch.flatten())
        
        return np.array(patches)
    
    def _forward(self, x):
        """Forward pass through the autoencoder."""
        # Encoder
        h1 = self._relu(np.dot(x, self.W_encoder) + self.b_encoder)
        latent = self._sigmoid(np.dot(h1, self.W_latent) + self.b_latent)
        
        # Decoder  
        h2 = self._relu(np.dot(latent, self.W_decode) + self.b_decode)
        output = self._sigmoid(np.dot(h2, self.W_output) + self.b_output)
        
        return output, latent, h1, h2
    
    def train(self, training_images, n_epochs=50, learning_rate=0.01):
        """
        Train the autoencoder on patches from training images.
        
        Args:
            training_images: List of 2D numpy arrays representing training scenes
            n_epochs: Number of training epochs (default: 50)
            learning_rate: Learning rate for gradient descent (default: 0.01)
        """
        # Extract training patches from all images
        all_patches = []
        for img in training_images:
            patches = self._extract_patches(img, n_patches=10)
            all_patches.append(patches)
        
        X_train = np.vstack(all_patches)
        
        print(f"Training autoencoder on {X_train.shape[0]} patches...")
        
        # Training loop
        for epoch in range(n_epochs):
            total_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            
            for i in range(0, len(X_shuffled), 32):  # Mini-batch size of 32
                batch = X_shuffled[i:i+32]
                
                # Add noise for denoising
                noisy_batch = batch + self.noise_factor * np.random.randn(*batch.shape)
                noisy_batch = np.clip(noisy_batch, 0, 1)
                
                # Forward pass
                output, latent, h1, h2 = self._forward(noisy_batch)
                
                # Compute loss (MSE)
                loss = np.mean((output - batch) ** 2)
                total_loss += loss
                
                # Backward pass (simplified gradient descent)
                # Output layer gradients
                d_output = 2 * (output - batch) / batch.shape[0]
                d_W_output = np.dot(h2.T, d_output)
                d_b_output = np.sum(d_output, axis=0)
                
                # Hidden layer 2 gradients
                d_h2 = np.dot(d_output, self.W_output.T)
                d_h2[h2 <= 0] = 0  # ReLU derivative
                d_W_decode = np.dot(latent.T, d_h2)
                d_b_decode = np.sum(d_h2, axis=0)
                
                # Latent layer gradients
                d_latent = np.dot(d_h2, self.W_decode.T)
                d_latent = d_latent * latent * (1 - latent)  # Sigmoid derivative
                d_W_latent = np.dot(h1.T, d_latent)
                d_b_latent = np.sum(d_latent, axis=0)
                
                # Hidden layer 1 gradients
                d_h1 = np.dot(d_latent, self.W_latent.T)
                d_h1[h1 <= 0] = 0  # ReLU derivative
                d_W_encoder = np.dot(noisy_batch.T, d_h1)
                d_b_encoder = np.sum(d_h1, axis=0)
                
                # Update weights
                self.W_output -= learning_rate * d_W_output
                self.b_output -= learning_rate * d_b_output
                self.W_decode -= learning_rate * d_W_decode
                self.b_decode -= learning_rate * d_b_decode
                self.W_latent -= learning_rate * d_W_latent
                self.b_latent -= learning_rate * d_b_latent
                self.W_encoder -= learning_rate * d_W_encoder
                self.b_encoder -= learning_rate * d_b_encoder
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(X_shuffled) // 32 + 1)
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        self.is_trained = True
        print("Autoencoder training completed!")
    
    def encode_patches(self, img, n_patches=8):
        """
        Extract and encode patches from an image.
        
        Args:
            img: Input image as 2D numpy array
            n_patches: Number of patches to extract and average (default: 8)
            
        Returns:
            numpy.ndarray: Average latent representation of patches
        """
        if not self.is_trained:
            # Use random encoding if not trained
            patches = self._extract_patches(img, n_patches)
            return np.random.rand(self.latent_dim) * 0.1
        
        patches = self._extract_patches(img, n_patches)
        _, latents, _, _ = self._forward(patches)
        
        # Return average latent representation
        return np.mean(latents, axis=0)
    
    def save(self, filepath):
        """Save the trained autoencoder to disk."""
        model_data = {
            'W_encoder': self.W_encoder,
            'b_encoder': self.b_encoder,
            'W_latent': self.W_latent,
            'b_latent': self.b_latent,
            'W_decode': self.W_decode,
            'b_decode': self.b_decode,
            'W_output': self.W_output,
            'b_output': self.b_output,
            'patch_size': self.patch_size,
            'latent_dim': self.latent_dim,
            'noise_factor': self.noise_factor,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load a trained autoencoder from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W_encoder = model_data['W_encoder']
        self.b_encoder = model_data['b_encoder']
        self.W_latent = model_data['W_latent']
        self.b_latent = model_data['b_latent']
        self.W_decode = model_data['W_decode']
        self.b_decode = model_data['b_decode']
        self.W_output = model_data['W_output']
        self.b_output = model_data['b_output']
        self.patch_size = model_data['patch_size']
        self.latent_dim = model_data['latent_dim']
        self.noise_factor = model_data['noise_factor']
        self.is_trained = model_data['is_trained']


# Global autoencoder instance for terminal features
_global_autoencoder = None


def get_autoencoder(retrain=False):
    """
    Get or create the global autoencoder instance.
    
    Args:
        retrain: Whether to retrain the autoencoder (default: False)
        
    Returns:
        SimpleAutoencoder: The global autoencoder instance
    """
    global _global_autoencoder
    
    model_path = '/tmp/recon_autoencoder.pkl'
    
    if _global_autoencoder is None:
        _global_autoencoder = SimpleAutoencoder(patch_size=8, latent_dim=4)
        
        # Try to load existing trained model
        if os.path.exists(model_path) and not retrain:
            try:
                _global_autoencoder.load(model_path)
                print("Loaded pretrained autoencoder")
            except:
                print("Failed to load autoencoder, will train new one")
                _global_autoencoder.is_trained = False
        
        # Train if needed and enabled via env var
        train_enabled = os.environ.get('RECON_TRAIN_AE', '0') in ('1','true','True')
        if (not _global_autoencoder.is_trained or retrain) and train_enabled:
            from .dataset import make_house_scene, make_barn_scene, make_varied_scene
            
            print("Training autoencoder...")
            # Generate diverse training data
            training_images = []
            for _ in range(20):
                training_images.append(make_house_scene(noise=0.05))
                training_images.append(make_barn_scene(noise=0.05))
                training_images.append(make_varied_scene('house', noise=0.1))
                training_images.append(make_varied_scene('barn', noise=0.1))
            
            _global_autoencoder.train(training_images, n_epochs=30)
            _global_autoencoder.save(model_path)
    
    return _global_autoencoder


def autoencoder_terminals_from_image(img):
    """
    Extract autoencoder-based terminal features from an image.
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary mapping autoencoder terminal IDs to activation values
    """
    ae = get_autoencoder()
    
    # Get latent representation
    latent_features = ae.encode_patches(img, n_patches=12)
    
    # Create terminal activations from latent features
    terminals = {}
    for i, feat_val in enumerate(latent_features):
        terminals[f't_ae_{i}'] = float(feat_val)
    
    return terminals


def comprehensive_terminals_from_image(img):
    """
    Extract comprehensive terminal features using all available methods.
    
    Combines basic filters, advanced features, and autoencoder features
    into one comprehensive feature set for maximum representational power.
    
    Args:
        img: Input image as 2D numpy array
        
    Returns:
        dict: Dictionary mapping all terminal IDs to activation values
    """
    # Get all feature types
    advanced_features = advanced_terminals_from_image(img)
    autoencoder_features = autoencoder_terminals_from_image(img)
    
    # Combine all features
    all_terminals = {}
    all_terminals.update(advanced_features)
    all_terminals.update(autoencoder_features)
    
    return all_terminals


def comprehensive_sample_scene_and_terminals():
    """
    Generate a random synthetic scene and extract all available terminal features.

    Returns:
        tuple: (image, terminals) where:
               - image: Generated 2D numpy array representing the scene
               - terminals: Dictionary of all available terminal unit activations
    """
    img = make_house_scene()
    t = comprehensive_terminals_from_image(img)
    return img, t
