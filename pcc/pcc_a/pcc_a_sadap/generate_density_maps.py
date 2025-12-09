#!/usr/bin/env python3
"""
Density Map Generation Script for PCC-Net
Converts point annotations to density maps using Gaussian kernel approach
Based on get_density_map_gaussian.m MATLAB script logic
"""

import os
import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree
from PIL import Image
import argparse
from tqdm import tqdm
import glob

def create_gaussian_kernel(rows: int, cols: int, sigma: float) -> np.ndarray:
    """Return a 2D Gaussian normalized to sum=1 with given shape (rows, cols)."""
    rows = int(rows); cols = int(cols)
    cy = rows // 2
    cx = cols // 2
    y, x = np.mgrid[-cy:rows-cy, -cx:cols-cx]
    H = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
    s = H.sum()
    if s > 0:
        H = H / s
    return H.astype(np.float32)

def _odd(n: int) -> int:
    return n if (n % 2) == 1 else n + 1

def _round_to(x: float, step: float) -> float:
    if step and step > 0:
        return float(np.round(x / step) * step)
    return float(x)

def get_density_map_gaussian(
    image_shape,
    points,
    *,
    # Adaptive KNN hyperparameters (tuned to match your B script defaults)
    knn_k: int = 3,               # neighbors to aggregate (excluding self)
    knn_beta: float = 0.1,        # scale factor for sigma from summed distances
    sigma_min: float = 1.0,       # clamp lower bound
    sigma_max: float = 25.0,      # clamp upper bound
    ksize_factor: float = 3.0,    # k ≈ 2*ceil(ksize_factor*σ)+1
    ksize_min: int = 7,           # min odd kernel size
    ksize_max: int = 61,          # max odd kernel size
    sigma_round: float = 0.25     # round σ for kernel cache key (0 disables)
) -> np.ndarray:
    """
    Generate density map from point annotations using adaptive KNN Gaussian kernels.
    Always computes per-point sigma using K-nearest neighbor distances.
    
    Args:
        image_shape: (height, width) of the target image
        points: Nx2 array of (x, y) coordinates
        knn_k: neighbors to aggregate (excluding self)
        knn_beta: scale factor for sigma from summed distances
        sigma_min: clamp lower bound
        sigma_max: clamp upper bound
        ksize_factor: k ≈ 2*ceil(ksize_factor*σ)+1
        ksize_min: min odd kernel size
        ksize_max: max odd kernel size
        sigma_round: round σ for kernel cache key (0 disables)
    
    Returns:
        density_map: 2D numpy array with density values
    """
    h, w = image_shape
    im_density = np.zeros((h, w), dtype=np.float32)

    N = len(points)
    if N == 0:
        return im_density

    # Build KDTree only if we have >= 2 points
    if N >= 2:
        # +1 because query returns self at dist 0
        kq = min(knn_k + 1, N)
        tree = KDTree(points)
        dists, _ = tree.query(points, k=kq)
        if dists.ndim == 1:
            dists = dists[:, None]
    else:
        dists = None

    # Cache full, unclipped kernels by (sigma_quantized, ksize)
    kernel_cache: dict[tuple[float, int], np.ndarray] = {}

    def compute_sigma_i(i: int) -> float:
        # Sum distances to the nearest knn_k neighbors (exclude self)
        if dists is not None and dists.shape[1] >= 2:
            neigh = dists[i][1:1+min(knn_k, dists.shape[1]-1)]
            s = float(neigh.sum())
            sigma_i = knn_beta * s if s > 0 else 0.0
        else:
            # Singleton fallback: scale to image size so we still place a Gaussian
            sigma_i = knn_beta * 0.02 * float(np.hypot(h, w))
        sigma_i = max(sigma_min, min(sigma_max, sigma_i))
        return _round_to(sigma_i, sigma_round) if sigma_round and sigma_round > 0 else sigma_i

    def compute_ksize_i(sigma_i: float) -> int:
        sz = 2 * int(np.ceil(ksize_factor * sigma_i)) + 1
        sz = max(ksize_min, min(ksize_max, sz))
        return _odd(sz)

    # Always place Gaussians (even for single point)
    for i in range(N):
        sigma_i = compute_sigma_i(i)
        ksize = compute_ksize_i(sigma_i)

        # MATLAB-parity coordinate handling: floor -> abs -> clamp
        x = int(np.floor(points[i, 0]));  x = abs(x);  x = max(0, min(w - 1, x))
        y = int(np.floor(points[i, 1]));  y = abs(y);  y = max(0, min(h - 1, y))

        x1 = x - ksize // 2; y1 = y - ksize // 2
        x2 = x + ksize // 2; y2 = y + ksize // 2

        change_H = False
        if x1 < 0:  x1 = 0;       change_H = True
        if y1 < 0:  y1 = 0;       change_H = True
        if x2 >= w: x2 = w - 1;   change_H = True
        if y2 >= h: y2 = h - 1;   change_H = True

        if change_H:
            roi_rows = (y2 - y1 + 1); roi_cols = (x2 - x1 + 1)
            H_adjusted = create_gaussian_kernel(roi_rows, roi_cols, sigma_i)
        else:
            key = (sigma_i, ksize)
            H_adjusted = kernel_cache.get(key)
            if H_adjusted is None:
                H_adjusted = create_gaussian_kernel(ksize, ksize, sigma_i)
                kernel_cache[key] = H_adjusted

        # Safety: ensure ROI matches kernel shape
        assert H_adjusted.shape == (y2 - y1 + 1, x2 - x1 + 1)
        im_density[y1:y2+1, x1:x2+1] += H_adjusted

    return im_density

def process_dataset(input_dir, output_dir, target_size=(576, 768)):
    """
    Process the entire dataset: generate adaptive KNN density maps from point annotations
    
    Args:
        input_dir: path to input dataset directory (part_A_converted)
        output_dir: path to output dataset directory (same as input for now)
        target_size: (height, width) of target image size
    """
    
    # Define paths
    train_img_dir = os.path.join(input_dir, 'train_data', 'images')
    train_gt_dir = os.path.join(input_dir, 'train_data', 'ground_truth')
    train_den_dir = os.path.join(input_dir, 'train_data', 'den')
    
    test_img_dir = os.path.join(input_dir, 'test_data', 'images')
    test_gt_dir = os.path.join(input_dir, 'test_data', 'ground_truth')
    test_den_dir = os.path.join(input_dir, 'test_data', 'den')
    
    # Create output directories
    os.makedirs(train_den_dir, exist_ok=True)
    os.makedirs(test_den_dir, exist_ok=True)
    
    print(f"Processing dataset: {input_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]} (height x width)")
    print(f"Output density maps: {train_den_dir} and {test_den_dir}")
    
    # Process training data
    print("\nProcessing training data...")
    train_files = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    
    for img_path in tqdm(train_files, desc="Training data"):
        # Get corresponding ground truth file
        img_name = os.path.basename(img_path)
        gt_name = img_name.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        gt_path = os.path.join(train_gt_dir, gt_name)
        
        if os.path.exists(gt_path):
            # Load point annotations
            mat_data = sio.loadmat(gt_path)
            points = mat_data['annPoints']  # Extract coordinates
            
            # Load image to get actual dimensions
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Use actual image dimensions (like MATLAB's size(im))
            H_img, W_img = img_array.shape[:2]
            
            # Generate density map using actual image size
            density_map = get_density_map_gaussian((H_img, W_img), points)
            
            # Save as CSV
            csv_name = img_name.replace('.jpg', '.csv')
            csv_path = os.path.join(train_den_dir, csv_name)
            np.savetxt(csv_path, density_map, delimiter=',', fmt='%.8f')
            
        else:
            print(f"Warning: Ground truth file not found for {img_name}")
    
    # Process test data
    print("\nProcessing test data...")
    test_files = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    
    for img_path in tqdm(test_files, desc="Test data"):
        # Get corresponding ground truth file
        img_name = os.path.basename(img_path)
        gt_name = img_name.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        gt_path = os.path.join(test_gt_dir, gt_name)
        
        if os.path.exists(gt_path):
            # Load point annotations
            mat_data = sio.loadmat(gt_path)
            points = mat_data['annPoints']  # Extract coordinates
            
            # Load image to get actual dimensions
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Use actual image dimensions (like MATLAB's size(im))
            H_img, W_img = img_array.shape[:2]
            
            # Generate density map using actual image size
            density_map = get_density_map_gaussian((H_img, W_img), points)
            
            # Save as CSV
            csv_name = img_name.replace('.jpg', '.csv')
            csv_path = os.path.join(test_den_dir, csv_name)
            np.savetxt(csv_path, density_map, delimiter=',', fmt='%.8f')
            
        else:
            print(f"Warning: Ground truth file not found for {img_name}")
    
    print(f"\nDensity map generation completed!")
    print(f"Training density maps saved to: {train_den_dir}")
    print(f"Test density maps saved to: {test_den_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate density maps from point annotations')
    parser.add_argument('--input_dir', type=str, default='part_A_converted', 
                        help='Input dataset directory (part_A_converted)')
    parser.add_argument('--target_height', type=int, default=576, 
                        help='Target image height')
    parser.add_argument('--target_width', type=int, default=768, 
                        help='Target image width')
    
    args = parser.parse_args()
    
    target_size = (args.target_height, args.target_width)
    
    print("="*60)
    print("Adaptive KNN Density Map Generation for PCC-Net")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print("Generating adaptive KNN density maps (fixed mode removed)")
    print("="*60)
    
    # Process dataset
    process_dataset(args.input_dir, args.input_dir, target_size)

if __name__ == "__main__":
    main()
