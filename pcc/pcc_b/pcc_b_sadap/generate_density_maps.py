#!/usr/bin/env python3
"""
Density Map Generation Script for PCC-Net
Converts point annotations to density maps using Gaussian kernel approach
Based on get_density_map_gaussian.m MATLAB script logic
"""

import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from PIL import Image
import argparse
from tqdm import tqdm
import glob

def _odd(n: int) -> int:
    """Ensure integer is odd"""
    return n if n % 2 == 1 else n + 1

def _round_to(x: float, step: float) -> float:
    """Round x to nearest step for caching"""
    if step <= 0: 
        return float(x)
    return round(x / step) * step

def create_gaussian_kernel(rows, cols, sigma):
    """
    Create a 2D Gaussian kernel with specified dimensions
    Equivalent to MATLAB's fspecial('Gaussian', [rows, cols], sigma)
    """
    # Create coordinate grids
    x = np.arange(cols) - cols // 2
    y = np.arange(rows) - rows // 2
    X, Y = np.meshgrid(x, y)
    
    # Calculate Gaussian
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    return kernel

def get_density_map_gaussian(
    image_shape,
    points,
    f_sz=15,
    sigma=15.0,
    sigma_mode: str = "fixed",         # "fixed" (current) or "adaptive_knn"
    knn_k: int = 3,                    # nearest neighbors to sum
    knn_beta: float = 0.1,             # β scale for sigma = β*sum(dists)
    sigma_min: float = 1.0,            # clamp to avoid too tiny kernels
    sigma_max: float = 25.0,           # clamp to avoid huge kernels
    ksize_factor: float = 3.0,         # ksize = 2*ceil(factor*sigma)+1
    ksize_min: int = 7,                # lower bound on kernel size (odd)
    ksize_max: int = 61,               # upper bound on kernel size (odd)
    sigma_round: float = 0.25          # cache key granularity for kernels
):
    """
    Generate density map from point annotations using Gaussian kernels
    Based on get_density_map_gaussian.m MATLAB script
    
    Args:
        image_shape: (height, width) of the target image
        points: Nx2 array of (x, y) coordinates
        f_sz: Gaussian kernel size (default: 15)
        sigma: Gaussian sigma (default: 4.0)
    
    Returns:
        density_map: 2D numpy array with density values
    """
    h, w = image_shape
    im_density = np.zeros((h, w), dtype=np.float32)
    
    # Early exits (preserve current behavior for fixed mode)
    if len(points) == 0:
        return im_density

    if sigma_mode == "fixed" and len(points) == 1:
        # preserve original single-point impulse behavior in fixed mode
        x1 = max(0, min(w-1, int(round(points[0, 0]))))
        y1 = max(0, min(h-1, int(round(points[0, 1]))))
        im_density[y1, x1] = 255.0
        return im_density

    # Prepare cache for kernels across (sigma, ksize)
    kernel_cache = {}  # key: (float_sigma_quantized, int_ksize) -> np.ndarray

    # Precompute fixed-mode base kernel (unchanged)
    H_fixed = None
    if sigma_mode == "fixed":
        H_fixed = create_gaussian_kernel(f_sz, f_sz, sigma)

    # Build per-point sigma for adaptive_knn
    sigma_arr = None
    if sigma_mode == "adaptive_knn":
        if len(points) == 1:
            # single-head Gaussian (different from fixed-mode impulse)
            sigma_single = 0.25 * ((h + w) / 2.0)  # (avg(H,W))/4  => (H+W)/8
            sigma_single = float(np.clip(sigma_single, sigma_min, sigma_max))
            sigma_arr = np.array([sigma_single], dtype=np.float32)
        else:
            # KD-Tree on (x, y) coordinates
            pts = np.asarray(points, dtype=np.float32)
            tree = KDTree(pts)
            K = min(knn_k + 1, len(pts))  # include self; we'll drop it
            dists, idxs = tree.query(pts, k=K)
            # Drop self distance at [:,0]; sum next knn_k distances
            # If K could be < (knn_k+1), handle with nan_to_num
            if K > 1:
                dsum = np.sum(dists[:, 1:K], axis=1)
            else:
                dsum = np.zeros((len(pts),), dtype=np.float32)
            sigma_arr = knn_beta * dsum
            # clamp
            sigma_arr = np.clip(sigma_arr, sigma_min, sigma_max).astype(np.float32)
    
    for j in range(len(points)):
        # MATLAB parity: abs(int32(floor(point)))
        x = int(np.floor(points[j, 0]));  x = abs(x);  x = max(0, min(w-1, x))
        y = int(np.floor(points[j, 1]));  y = abs(y);  y = max(0, min(h-1, y))
        
        # Choose sigma and kernel size for this point
        if sigma_mode == "adaptive_knn":
            sigma_j = float(sigma_arr[j])
            # dynamic kernel size that scales with sigma
            k_half = int(np.ceil(ksize_factor * sigma_j))
            ksize = _odd(2 * k_half + 1)
            ksize = int(np.clip(ksize, ksize_min, ksize_max))
        else:
            sigma_j = float(sigma)          # fixed
            ksize = int(f_sz)
            k_half = ksize // 2

        # Compute placement window from the (possibly dynamic) half-size
        x1 = x - k_half; y1 = y - k_half
        x2 = x + k_half; y2 = y + k_half

        dfx1 = dfy1 = dfx2 = dfy2 = 0
        change_H = False

        if x1 < 0:
            dfx1 = -x1; x1 = 0; change_H = True
        if y1 < 0:
            dfy1 = -y1; y1 = 0; change_H = True
        if x2 >= w:
            dfx2 = x2 - (w - 1); x2 = w - 1; change_H = True
        if y2 >= h:
            dfy2 = y2 - (h - 1); y2 = h - 1; change_H = True

        # Choose / build the kernel for this point
        roi_rows = (y2 - y1 + 1)
        roi_cols = (x2 - x1 + 1)
        
        if change_H or (sigma_mode == "adaptive_knn" and (roi_rows != ksize or roi_cols != ksize)):
            # Always regenerate kernel when ROI size doesn't match expected kernel size
            H_current = create_gaussian_kernel(roi_rows, roi_cols, sigma_j)
        else:
            if sigma_mode == "fixed":
                H_current = H_fixed
            else:
                # quantize sigma for cache key to avoid blow-up
                s_key = _round_to(sigma_j, sigma_round)
                key = (float(s_key), int(ksize))
                H_current = kernel_cache.get(key)
                if H_current is None:
                    H_current = create_gaussian_kernel(ksize, ksize, float(s_key))
                    kernel_cache[key] = H_current

        # Safety & accumulation (unchanged)
        assert H_current.shape == (y2 - y1 + 1, x2 - x1 + 1)
        im_density[y1:y2+1, x1:x2+1] += H_current
    
    return im_density

def process_dataset(input_dir, output_dir, target_size=(576, 768), args=None):
    """
    Process the entire dataset: generate density maps from point annotations
    
    Args:
        input_dir: path to input dataset directory (part_B_converted)
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
            density_map = get_density_map_gaussian(
                (H_img, W_img), points,
                f_sz=15, sigma=15.0,
                sigma_mode=args.sigma_mode,
                knn_k=args.knn_k,
                knn_beta=args.knn_beta,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                ksize_factor=args.ksize_factor,
                ksize_min=args.ksize_min,
                ksize_max=args.ksize_max,
                sigma_round=args.sigma_round
            )
            
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
            density_map = get_density_map_gaussian(
                (H_img, W_img), points,
                f_sz=15, sigma=15.0,
                sigma_mode=args.sigma_mode,
                knn_k=args.knn_k,
                knn_beta=args.knn_beta,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                ksize_factor=args.ksize_factor,
                ksize_min=args.ksize_min,
                ksize_max=args.ksize_max,
                sigma_round=args.sigma_round
            )
            
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
    parser.add_argument('--input_dir', type=str, default='part_B_converted', 
                        help='Input dataset directory (part_B_converted)')
    parser.add_argument('--target_height', type=int, default=576, 
                        help='Target image height')
    parser.add_argument('--target_width', type=int, default=768, 
                        help='Target image width')
    parser.add_argument('--sigma_mode', type=str, default='fixed',
                        choices=['fixed', 'adaptive_knn'],
                        help="Sigma selection: 'fixed' (legacy) or 'adaptive_knn' (KD-Tree).")
    parser.add_argument('--knn_k', type=int, default=3, help='K nearest neighbors for sigma.')
    parser.add_argument('--knn_beta', type=float, default=0.1, help='Beta scale for sigma.')
    parser.add_argument('--sigma_min', type=float, default=1.0, help='Lower clamp for sigma.')
    parser.add_argument('--sigma_max', type=float, default=25.0, help='Upper clamp for sigma.')
    parser.add_argument('--ksize_factor', type=float, default=3.0, help='Kernel half-size ≈ factor*sigma.')
    parser.add_argument('--ksize_min', type=int, default=7, help='Minimum odd kernel size.')
    parser.add_argument('--ksize_max', type=int, default=61, help='Maximum odd kernel size.')
    parser.add_argument('--sigma_round', type=float, default=0.25, help='Round sigma for caching.')
    
    args = parser.parse_args()
    
    target_size = (args.target_height, args.target_width)
    
    print("="*60)
    print("Density Map Generation for PCC-Net")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print("="*60)
    
    # Process dataset
    process_dataset(args.input_dir, args.input_dir, target_size, args)

if __name__ == "__main__":
    main()
