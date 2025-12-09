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
from PIL import Image
import argparse
from tqdm import tqdm
import glob

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

def get_density_map_gaussian(image_shape, points, f_sz=15, sigma=15.0):
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
    
    if len(points) == 0:
        return im_density
    
    if len(points) == 1:
        x1 = max(0, min(w-1, int(round(points[0, 0]))))
        y1 = max(0, min(h-1, int(round(points[0, 1]))))
        im_density[y1, x1] = 255.0
        return im_density
    
    # Create base Gaussian kernel once (f_sz x f_sz)
    H = create_gaussian_kernel(f_sz, f_sz, sigma)
    
    for j in range(len(points)):
        # MATLAB parity: abs(int32(floor(point)))
        x = int(np.floor(points[j, 0]));  x = abs(x);  x = max(0, min(w-1, x))
        y = int(np.floor(points[j, 1]));  y = abs(y);  y = max(0, min(h-1, y))
        
        # Calculate kernel placement boundaries
        x1 = x - f_sz // 2
        y1 = y - f_sz // 2
        x2 = x + f_sz // 2
        y2 = y + f_sz // 2
        
        # Initialize boundary adjustment variables
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        
        # Handle boundary cases
        if x1 < 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        
        if y1 < 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        
        if x2 >= w:
            dfx2 = x2 - w + 1
            x2 = w - 1
            change_H = True
        
        if y2 >= h:
            dfy2 = y2 - h + 1
            y2 = h - 1
            change_H = True
        
        if change_H:
            # Compute resulting ROI size
            roi_rows = (y2 - y1 + 1)
            roi_cols = (x2 - x1 + 1)

            # Regenerate a rectangular Gaussian kernel (normalized)
            H_adjusted = create_gaussian_kernel(roi_rows, roi_cols, sigma)
        else:
            H_adjusted = H
        
        # Shape assertion before adding (safety check)
        assert H_adjusted.shape == (y2 - y1 + 1, x2 - x1 + 1)
        im_density[y1:y2+1, x1:x2+1] += H_adjusted
    
    return im_density

def process_dataset(input_dir, output_dir, target_size=(576, 768)):
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
    parser.add_argument('--input_dir', type=str, default='part_B_converted', 
                        help='Input dataset directory (part_B_converted)')
    parser.add_argument('--target_height', type=int, default=576, 
                        help='Target image height')
    parser.add_argument('--target_width', type=int, default=768, 
                        help='Target image width')
    
    args = parser.parse_args()
    
    target_size = (args.target_height, args.target_width)
    
    print("="*60)
    print("Density Map Generation for PCC-Net")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print("="*60)
    
    # Process dataset
    process_dataset(args.input_dir, args.input_dir, target_size)

if __name__ == "__main__":
    main()
