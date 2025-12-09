#!/usr/bin/env python3
"""
Dataset Conversion Script for PCC-Net - Clean Head Coordinates
Converts images and scales head coordinates to target resolution

Outputs clean Nx2 coordinate arrays in annPoints format for MATLAB compatibility
"""

import os
import numpy as np
from PIL import Image
import scipy.io as sio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def resize_image_and_points(image_path, points, target_size=(768, 576)):
    """
    Resize image and scale point coordinates accordingly
    
    Args:
        image_path: path to the image file
        points: numpy array of shape (N, 2) containing (x, y) coordinates (PIL format)
        target_size: (width, height) of target image size (PIL format)
    
    Returns:
        resized_image: PIL Image object
        scaled_points: numpy array of scaled point coordinates (PIL format)
        scale_factors: (scale_x, scale_y) used for scaling
    """
    # Load image
    image = Image.open(image_path)
    original_size = image.size  # (width, height) - PIL format
    
    # Calculate scale factors
    scale_x = target_size[0] / original_size[0]  # width scale
    scale_y = target_size[1] / original_size[1]  # height scale
    
    # Resize image
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # Scale point coordinates
    if len(points) > 0:
        scaled_points = points.copy()
        scaled_points[:, 0] *= scale_x  # scale x coordinates
        scaled_points[:, 1] *= scale_y  # scale y coordinates
        
        # Clip coordinates to ensure they're within bounds
        scaled_points[:, 0] = np.clip(scaled_points[:, 0], 0, target_size[0] - 1)
        scaled_points[:, 1] = np.clip(scaled_points[:, 1], 0, target_size[1] - 1)
    else:
        scaled_points = points
    
    return resized_image, scaled_points, (scale_x, scale_y)

def process_dataset(input_dir, output_dir, target_size=(768, 576)):
    """
    Process the entire dataset: resize images and scale head coordinates
    
    Args:
        input_dir: path to input dataset directory
        output_dir: path to output dataset directory
        target_size: (width, height) of target image size (PIL format)
    """
    
    # Create output directory structure
    train_img_dir = os.path.join(output_dir, 'train_data', 'images')
    train_den_dir = os.path.join(output_dir, 'train_data', 'ground_truth')
    test_img_dir = os.path.join(output_dir, 'test_data', 'images')
    test_den_dir = os.path.join(output_dir, 'test_data', 'ground_truth')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_den_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_den_dir, exist_ok=True)
    
    print(f"Created output directory structure in: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]} (width x height)")
    print(f"Output: Clean head coordinates (annPoints format)")
    
    # Process training data
    print("\nProcessing training data...")
    train_img_path = os.path.join(input_dir, 'train_data', 'images')
    train_gt_path = os.path.join(input_dir, 'train_data', 'ground_truth')
    
    train_files = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
    
    for filename in tqdm(train_files, desc="Training data"):
        # Process image
        img_path = os.path.join(train_img_path, filename)
        
        # Load annotations
        gt_filename = filename.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        gt_path = os.path.join(train_gt_path, gt_filename)
        
        if os.path.exists(gt_path):
            # Load point annotations
            mat_data = sio.loadmat(gt_path)
            points = mat_data['image_info'][0, 0]['location'][0, 0]
            # Convert to 0-based indexing (MATLAB is 1-based, Python is 0-based)
            points = points.astype(np.float32) - 1.0
            
            # Resize image and scale points
            resized_img, scaled_points, scale_factors = resize_image_and_points(
                img_path, points, target_size
            )
            
            # Save resized image
            output_img_path = os.path.join(train_img_dir, filename)
            resized_img.save(output_img_path)
            
            # Save scaled head coordinates as clean Nx2 array
            output_gt_path = os.path.join(train_den_dir, gt_filename)
            sio.savemat(output_gt_path, {'annPoints': scaled_points.astype('float32')})
            
        else:
            print(f"Warning: Ground truth file not found for {filename}")
    
    # Process test data
    print("\nProcessing test data...")
    test_img_path = os.path.join(input_dir, 'test_data', 'images')
    test_gt_path = os.path.join(input_dir, 'test_data', 'ground_truth')
    
    test_files = [f for f in os.listdir(test_img_path) if f.endswith('.jpg')]
    
    for filename in tqdm(test_files, desc="Test data"):
        # Process image
        img_path = os.path.join(test_img_path, filename)
        
        # Load annotations
        gt_filename = filename.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        gt_path = os.path.join(test_gt_path, gt_filename)
        
        if os.path.exists(gt_path):
            # Load point annotations
            mat_data = sio.loadmat(gt_path)
            points = mat_data['image_info'][0, 0]['location'][0, 0]
            # Convert to 0-based indexing (MATLAB is 1-based, Python is 0-based)
            points = points.astype(np.float32) - 1.0
            
            # Resize image and scale points
            resized_img, scaled_points, scale_factors = resize_image_and_points(
                img_path, points, target_size
            )
            
            # Save resized image
            output_img_path = os.path.join(test_img_dir, filename)
            resized_img.save(output_img_path)
            
            # Save scaled head coordinates as clean Nx2 array
            output_gt_path = os.path.join(test_den_dir, gt_filename)
            sio.savemat(output_gt_path, {'annPoints': scaled_points.astype('float32')})
            
        else:
            print(f"Warning: Ground truth file not found for {filename}")
    
    print(f"\nDataset conversion completed!")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Saved clean head coordinates (annPoints format)")

def visualize_sample(input_dir, output_dir, sample_idx=1):
    """
    Visualize a sample conversion for verification
    
    Args:
        input_dir: path to input dataset directory
        output_dir: path to output dataset directory
        sample_idx: index of sample to visualize
    """
    # Load original data
    train_img_path = os.path.join(input_dir, 'train_data', 'images')
    train_gt_path = os.path.join(input_dir, 'train_data', 'ground_truth')
    
    img_files = sorted([f for f in os.listdir(train_img_path) if f.endswith('.jpg')])
    if sample_idx >= len(img_files):
        sample_idx = 0
    
    filename = img_files[sample_idx]
    img_path = os.path.join(train_img_path, filename)
    gt_filename = filename.replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
    gt_path = os.path.join(train_gt_path, gt_filename)
    
    # Load original image and points
    original_img = Image.open(img_path)
    mat_data = sio.loadmat(gt_path)
    original_points = mat_data['image_info'][0, 0]['location'][0, 0]
    # Convert to 0-based indexing for visualization
    original_points = original_points.astype(np.float32) - 1.0
    
    # Load converted data
    output_img_path = os.path.join(output_dir, 'train_data', 'images', filename)
    output_gt_path = os.path.join(output_dir, 'train_data', 'ground_truth', gt_filename)
    
    converted_img = Image.open(output_img_path)
    converted_data = sio.loadmat(output_gt_path)
    scaled_points = converted_data['annPoints']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image with points
    axes[0, 0].imshow(original_img)
    axes[0, 0].scatter(original_points[:, 0], original_points[:, 1], c='red', s=10, alpha=0.7)
    axes[0, 0].set_title(f'Original Image ({original_img.size[0]}x{original_img.size[1]})\nPoints: {len(original_points)}')
    axes[0, 0].axis('off')
    
    # Converted image with scaled points
    axes[0, 1].imshow(converted_img)
    axes[0, 1].scatter(scaled_points[:, 0], scaled_points[:, 1], c='red', s=10, alpha=0.7)
    axes[0, 1].set_title(f'Converted Image ({converted_img.size[0]}x{converted_img.size[1]})\nScaled Points: {len(scaled_points)}')
    axes[0, 1].axis('off')
    
    # Coordinate comparison plot
    axes[1, 0].scatter(original_points[:, 0], original_points[:, 1], c='blue', s=20, alpha=0.7, label='Original')
    axes[1, 0].scatter(scaled_points[:, 0], scaled_points[:, 1], c='red', s=20, alpha=0.7, label='Scaled')
    axes[1, 0].set_title('Coordinate Comparison\n(Blue: Original, Red: Scaled)')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Y coordinate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale factors and statistics
    scale_x = converted_img.size[0] / original_img.size[0]
    scale_y = converted_img.size[1] / original_img.size[1]
    axes[1, 1].text(0.1, 0.8, f'Scale Factors:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.7, f'X: {scale_x:.3f}', fontsize=10)
    axes[1, 1].text(0.1, 0.6, f'Y: {scale_y:.3f}', fontsize=10)
    axes[1, 1].text(0.1, 0.4, f'Point Count:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.3, f'Original: {len(original_points)}', fontsize=10)
    axes[1, 1].text(0.1, 0.2, f'Scaled: {len(scaled_points)}', fontsize=10)
    axes[1, 1].text(0.1, 0.05, f'Match: {"✓" if len(original_points) == len(scaled_points) else "✗"}', 
                   fontsize=12, fontweight='bold', color='green' if len(original_points) == len(scaled_points) else 'red')
    axes[1, 1].set_title('Conversion Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordinate_conversion_sample.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Coordinate conversion visualization saved as: {os.path.join(output_dir, 'coordinate_conversion_sample.png')}")
    print(f"Original points count: {len(original_points)}")
    print(f"Scaled points count: {len(scaled_points)}")
    print(f"Coordinate preservation: {'✓' if len(original_points) == len(scaled_points) else '✗'}")

def main():
    parser = argparse.ArgumentParser(description='Convert Shanghai Tech Part B dataset - Clean Head Coordinates')
    parser.add_argument('--input_dir', type=str, default='part_B', 
                        help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, default='part_B_converted', 
                        help='Output dataset directory')
    parser.add_argument('--target_width', type=int, default=768, 
                        help='Target image width')
    parser.add_argument('--target_height', type=int, default=576, 
                        help='Target image height')
    parser.add_argument('--visualize', action='store_true', 
                        help='Create visualization of coordinate conversion sample')
    
    args = parser.parse_args()
    
    target_size = (args.target_width, args.target_height)
    
    print("="*60)
    print("Shanghai Tech Part B Dataset Conversion - Clean Head Coordinates")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Output: Clean head coordinates (annPoints format)")
    print("="*60)
    
    # Process dataset
    process_dataset(args.input_dir, args.output_dir, target_size)
    
    # Create visualization if requested
    if args.visualize:
        print("\nCreating visualization...")
        visualize_sample(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
