import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

def generate_edge_map(image, method='canny', flatten=True):
    """
    Generate edge/shape map from image

    Args:
        image: PIL Image or numpy array (H, W, 3) with values 0-255
        method: 'canny' or 'sobel'
        flatten: if True, return flattened vector; if False, return 2D map

    Returns:
        Edge map as numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
    elif method == 'sobel':
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to 0-1
    edges = edges.astype(np.float32) / 255.0

    if flatten:
        # Resize to 16x16 and flatten to 256-dim vector
        edges = cv2.resize(edges, (16, 16))
        edges = edges.flatten()

    return edges

def shape_loss(pred_shape, gt_shape, loss_type='mse'):
    """
    Compute loss between predicted and ground truth shape maps

    Args:
        pred_shape: (batch, 256) predicted shape vectors
        gt_shape: (batch, 256) ground truth shape vectors
        loss_type: 'mse', 'bce', or 'l1'

    Returns:
        Loss value
    """
    if loss_type == 'mse':
        return F.mse_loss(pred_shape, gt_shape)
    elif loss_type == 'bce':
        return F.binary_cross_entropy(pred_shape, gt_shape)
    elif loss_type == 'l1':
        return F.l1_loss(pred_shape, gt_shape)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def batch_generate_edge_maps(images, method='canny'):
    """
    Generate edge maps for a batch of images

    Args:
        images: numpy array (batch, H, W, 3) with values 0-255
        method: edge detection method

    Returns:
        Edge maps as numpy array (batch, 256)
    """
    batch_size = len(images)
    edge_maps = np.zeros((batch_size, 256))

    for i, img in enumerate(images):
        edge_maps[i] = generate_edge_map(img, method=method, flatten=True)

    return edge_maps