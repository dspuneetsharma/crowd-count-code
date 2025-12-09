import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter  # Fixed import
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from torchvision import datasets, transforms
from scipy.ndimage import maximum_filter

# Use the same transform as training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = '../part_B'  # Fixed: Use correct relative path from CSRNet_adapted directory

#now generate the part_B's ground truth
part_B_train = os.path.join(root,'train_data','images')
part_B_test = os.path.join(root,'test_data','images')
path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

print(f"Found {len(img_paths)} test images")

model = CSRNet()
model = model.cuda()

# Load your trained model checkpoint - update path to match your training output
model_path = 'part_Bmodel_best.pth.tar'
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Best MAE during training: {checkpoint['best_prec1']:.6f}")
else:
    print(f"Error: Model file {model_path} not found!")
    print("Available model files:")
    for f in glob.glob("*.pth.tar"):
        print(f"  - {f}")
    exit(1)

model.eval()
mae = 0
mae_list = []
results_data = []  # Store all results for analysis

# Create output directories
output_dir = "validation_results"
predicted_maps_dir = os.path.join(output_dir, "predicted_density_maps")
density_png_dir = os.path.join(output_dir, "density_maps_png")
overlay_dir = os.path.join(output_dir, "overlay_images")
os.makedirs(predicted_maps_dir, exist_ok=True)
os.makedirs(density_png_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

print("\nTesting model on test images...")
print("="*60)
print(f"{'Image':<6} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12} {'Running MAE':<12}")
print("="*60)

def find_head_locations(density_map, threshold=0.1, min_distance=3):
    """Find head locations from density map using peak detection"""
    # Normalize density map
    density_norm = density_map / (density_map.max() + 1e-8)
    
    # Find local maxima using maximum filter approach
    # Create a mask where each pixel is the maximum in its neighborhood
    neighborhood_size = min_distance * 2 + 1
    max_filtered = maximum_filter(density_norm, size=neighborhood_size)
    
    # Find coordinates where the original equals the maximum (local maxima)
    local_maxima = (density_norm == max_filtered) & (density_norm > threshold)
    coordinates = np.column_stack(np.where(local_maxima))
    
    return coordinates

def create_overlay_image(original_img, density_map, head_coords, img_name):
    """Create overlay image showing original image, density map, and head locations"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image: {img_name}', fontsize=12)
    axes[0].axis('off')
    
    # Density map
    im1 = axes[1].imshow(density_map, cmap='jet')
    axes[1].set_title('Predicted Density Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay image with head locations
    axes[2].imshow(original_img)
    axes[2].set_title(f'Overlay: {len(head_coords)} heads detected', fontsize=12)
    axes[2].axis('off')
    
    # Plot head locations as red dots
    if len(head_coords) > 0:
        axes[2].scatter(head_coords[:, 1], head_coords[:, 0], 
                       c='red', s=20, alpha=0.8, marker='o')
    
    plt.tight_layout()
    return fig

for i in range(len(img_paths)):
    # Use the same preprocessing as training
    img = Image.open(img_paths[i]).convert('RGB')
    img_tensor = transform(img).cuda()
    
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
    
    pred_count = output.detach().cpu().sum().numpy()
    gt_count = np.sum(groundtruth)
    diff = abs(pred_count - gt_count)
    mae += diff
    mae_list.append(diff)
    
    # Store results for analysis
    img_name = os.path.basename(img_paths[i])
    results_data.append({
        'image_name': img_name,
        'image_path': img_paths[i],
        'predicted_count': pred_count,
        'ground_truth_count': gt_count,
        'difference': diff,
        'mae': diff
    })
    
    # Save predicted density map as H5
    pred_density = output.detach().cpu().squeeze().numpy()
    pred_filename = os.path.join(predicted_maps_dir, f"pred_{img_name.replace('.jpg', '.h5')}")
    with h5py.File(pred_filename, 'w') as hf:
        hf['density'] = pred_density
    
    # Save density map as PNG
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_density, cmap='jet')
    plt.colorbar(label='Density')
    plt.title(f'Predicted Density Map: {img_name}')
    plt.axis('off')
    png_filename = os.path.join(density_png_dir, f"density_{img_name.replace('.jpg', '.png')}")
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find head locations
    head_coordinates = find_head_locations(pred_density)
    
    # Create overlay image
    original_array = np.array(img)
    overlay_fig = create_overlay_image(original_array, pred_density, head_coordinates, img_name)
    overlay_filename = os.path.join(overlay_dir, f"overlay_{img_name.replace('.jpg', '.png')}")
    overlay_fig.savefig(overlay_filename, dpi=150, bbox_inches='tight')
    plt.close(overlay_fig)
    
    running_mae = mae / (i + 1)
    
    print(f"{i+1:<6} {pred_count:<12.2f} {gt_count:<12.2f} {diff:<12.2f} {running_mae:<12.2f}")

print("="*60)
final_mae = mae/len(img_paths)
print(f"\nFinal Results:")
print(f"Total test images: {len(img_paths)}")
print(f"Final MAE: {final_mae:.6f}")
print(f"Best MAE during training: {checkpoint['best_prec1']:.6f}")

# Calculate additional statistics
mae_array = np.array(mae_list)
print(f"MAE Statistics:")
print(f"  - Mean: {np.mean(mae_array):.6f}")
print(f"  - Std:  {np.std(mae_array):.6f}")
print(f"  - Min:  {np.min(mae_array):.6f}")
print(f"  - Max:  {np.max(mae_array):.6f}")

# Sort results by MAE (difference) to find best and worst predictions
results_data.sort(key=lambda x: x['difference'])

# Get 10 best and 10 worst predictions
best_10 = results_data[:10]
worst_10 = results_data[-10:]

# Save results to files
print(f"\nSaving results...")

# Save all results
all_results_file = os.path.join(output_dir, "all_validation_results.txt")
with open(all_results_file, 'w') as f:
    f.write("CSRNet Validation Results\n")
    f.write("="*50 + "\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Training Epoch: {checkpoint['epoch']}\n")
    f.write(f"Best Training MAE: {checkpoint['best_prec1']:.6f}\n")
    f.write(f"Total Test Images: {len(img_paths)}\n")
    f.write(f"Final Test MAE: {final_mae:.6f}\n")
    f.write(f"MAE Statistics:\n")
    f.write(f"  - Mean: {np.mean(mae_array):.6f}\n")
    f.write(f"  - Std:  {np.std(mae_array):.6f}\n")
    f.write(f"  - Min:  {np.min(mae_array):.6f}\n")
    f.write(f"  - Max:  {np.max(mae_array):.6f}\n\n")
    
    f.write("All Results (sorted by MAE):\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Image':<30} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12}\n")
    f.write("-"*50 + "\n")
    for result in results_data:
        f.write(f"{result['image_name']:<30} {result['predicted_count']:<12.2f} {result['ground_truth_count']:<12.2f} {result['difference']:<12.2f}\n")

# Save best and worst predictions
best_worst_file = os.path.join(output_dir, "best_worst_predictions.txt")
with open(best_worst_file, 'w') as f:
    f.write("CSRNet Best and Worst Predictions\n")
    f.write("="*50 + "\n\n")
    
    f.write("TOP 10 BEST PREDICTIONS (Lowest MAE):\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Rank':<5} {'Image':<30} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12}\n")
    f.write("-"*50 + "\n")
    for i, result in enumerate(best_10, 1):
        f.write(f"{i:<5} {result['image_name']:<30} {result['predicted_count']:<12.2f} {result['ground_truth_count']:<12.2f} {result['difference']:<12.2f}\n")
    
    f.write("\n" + "="*50 + "\n\n")
    
    f.write("TOP 10 WORST PREDICTIONS (Highest MAE):\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Rank':<5} {'Image':<30} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12}\n")
    f.write("-"*50 + "\n")
    for i, result in enumerate(reversed(worst_10), 1):
        f.write(f"{i:<5} {result['image_name']:<30} {result['predicted_count']:<12.2f} {result['ground_truth_count']:<12.2f} {result['difference']:<12.2f}\n")

# Save summary statistics
summary_file = os.path.join(output_dir, "validation_summary.txt")
with open(summary_file, 'w') as f:
    f.write("CSRNet Validation Summary\n")
    f.write("="*30 + "\n")
    f.write(f"Model File: {model_path}\n")
    f.write(f"Training Epoch: {checkpoint['epoch']}\n")
    f.write(f"Best Training MAE: {checkpoint['best_prec1']:.6f}\n")
    f.write(f"Total Test Images: {len(img_paths)}\n")
    f.write(f"Final Test MAE: {final_mae:.6f}\n")
    f.write(f"MAE Statistics:\n")
    f.write(f"  Mean: {np.mean(mae_array):.6f}\n")
    f.write(f"  Std:  {np.std(mae_array):.6f}\n")
    f.write(f"  Min:  {np.min(mae_array):.6f}\n")
    f.write(f"  Max:  {np.max(mae_array):.6f}\n")
    f.write(f"\nBest Prediction: {best_10[0]['image_name']} (MAE: {best_10[0]['difference']:.6f})\n")
    f.write(f"Worst Prediction: {worst_10[-1]['image_name']} (MAE: {worst_10[-1]['difference']:.6f}\n")

print(f"\nResults saved to:")
print(f"  - All results: {all_results_file}")
print(f"  - Best/Worst predictions: {best_worst_file}")
print(f"  - Summary: {summary_file}")
print(f"  - Predicted density maps (H5): {predicted_maps_dir}/")
print(f"  - Density maps (PNG): {density_png_dir}/")
print(f"  - Overlay images: {overlay_dir}/")
print(f"\nTop 3 Best Predictions:")
for i, result in enumerate(best_10[:3], 1):
    print(f"  {i}. {result['image_name']} - MAE: {result['difference']:.6f}")
print(f"\nTop 3 Worst Predictions:")
for i, result in enumerate(reversed(worst_10[:3]), 1):
    print(f"  {i}. {result['image_name']} - MAE: {result['difference']:.6f}")
