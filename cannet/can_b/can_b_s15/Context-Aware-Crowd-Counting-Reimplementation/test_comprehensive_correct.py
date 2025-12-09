import h5py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
from model import CANNet
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

def load_data(img_path, train=True):
    """Load and process image and ground truth exactly like in training"""
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])
    
    # Apply the same processing as in training (no augmentation for testing)
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    
    return img, target

def load_mat_annotations(mat_path):
    """Load original count from .mat file"""
    import scipy.io as io
    mat = io.loadmat(mat_path)
    # The count is in image_info[0,0]['number']
    count = mat['image_info'][0,0]['number'][0,0][0,0]
    return int(count)

def predict_density_map(model, img, device):
    """Predict density map using 4-quadrant inference"""
    # 4-quadrant inference (same as training validation)
    h, w = img.shape[2:4]
    h_d = h // 2
    w_d = w // 2
    
    img_1 = img[:, :, :h_d, :w_d]
    img_2 = img[:, :, :h_d, w_d:]
    img_3 = img[:, :, h_d:, :w_d]
    img_4 = img[:, :, h_d:, w_d:]
    
    with torch.no_grad():
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()
    
    # Calculate predicted count (same as training validation)
    pred_count = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
    
    # For visualization, combine quadrants and resize to original image size
    pred_density = np.zeros((h, w))
    
    # Resize each quadrant to match the target size
    density_1_resized = cv2.resize(density_1[0, 0, :, :], (w_d, h_d))
    density_2_resized = cv2.resize(density_2[0, 0, :, :], (w - w_d, h_d))
    density_3_resized = cv2.resize(density_3[0, 0, :, :], (w_d, h - h_d))
    density_4_resized = cv2.resize(density_4[0, 0, :, :], (w - w_d, h - h_d))
    
    pred_density[:h_d, :w_d] = density_1_resized
    pred_density[:h_d, w_d:] = density_2_resized
    pred_density[h_d:, :w_d] = density_3_resized
    pred_density[h_d:, w_d:] = density_4_resized
    
    return pred_density, pred_count

def save_density_map(density_map, save_path, title="Density Map"):
    """Save density map as image"""
    plt.figure(figsize=(8, 6))
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def combine_and_save_plots(original_img, gt_density, pred_density, save_path, 
                          gt_count, pred_count, original_count, mae):
    """Create combined visualization of original image, GT density, and predicted density"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image\nCount: {original_count}', fontsize=12)
    axes[0].axis('off')
    
    # Ground truth density
    im1 = axes[1].imshow(gt_density, cmap='jet')
    axes[1].set_title(f'Ground Truth Density\nCount: {gt_count:.1f}', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Predicted density
    im2 = axes[2].imshow(pred_density, cmap='jet')
    axes[2].set_title(f'Predicted Density\nCount: {pred_count:.1f}\nMAE: {mae:.2f}', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_to_excel(results, save_path):
    """Save detailed metrics to Excel file"""
    df = pd.DataFrame(results)
    df.to_excel(save_path, index=False)
    print(f"Detailed metrics saved to: {save_path}")

def save_best_worst_images(results, output_dir, num_images=10):
    """Save best and worst performing images with their details"""
    
    # Sort by MAE (ascending for best, descending for worst)
    sorted_results = sorted(results, key=lambda x: x['mae'])
    
    best_images = sorted_results[:num_images]
    worst_images = sorted_results[-num_images:]
    
    # Save best performing images
    best_dir = os.path.join(output_dir, 'best_performing')
    os.makedirs(best_dir, exist_ok=True)
    
    with open(os.path.join(best_dir, 'best_performing_images.txt'), 'w') as f:
        f.write("Top 10 Best Performing Images (Lowest MAE)\n")
        f.write("=" * 50 + "\n")
        for i, result in enumerate(best_images, 1):
            f.write(f"{i}. {result['image_name']}\n")
            f.write(f"   MAE: {result['mae']:.3f}\n")
            f.write(f"   Predicted Count: {result['predicted_count']:.1f}\n")
            f.write(f"   Ground Truth Count: {result['ground_truth_count']:.1f}\n")
            f.write(f"   Original Count: {result['original_count']}\n")
            f.write("\n")
    
    # Save worst performing images
    worst_dir = os.path.join(output_dir, 'worst_performing')
    os.makedirs(worst_dir, exist_ok=True)
    
    with open(os.path.join(worst_dir, 'worst_performing_images.txt'), 'w') as f:
        f.write("Top 10 Worst Performing Images (Highest MAE)\n")
        f.write("=" * 50 + "\n")
        for i, result in enumerate(worst_images, 1):
            f.write(f"{i}. {result['image_name']}\n")
            f.write(f"   MAE: {result['mae']:.3f}\n")
            f.write(f"   Predicted Count: {result['predicted_count']:.1f}\n")
            f.write(f"   Ground Truth Count: {result['ground_truth_count']:.1f}\n")
            f.write(f"   Original Count: {result['original_count']}\n")
            f.write("\n")

def test_model_comprehensive(model, test_json, device):
    """Comprehensive testing with all outputs"""
    
    # Load test data paths
    with open(test_json, 'r') as f:
        img_paths = json.load(f)
    
    # Transform for images (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create output directories
    output_dirs = [
        'outputs/test_results/density_maps/ground_truth',
        'outputs/test_results/density_maps/predicted', 
        'outputs/test_results/density_maps/combined',
        'outputs/test_results/best_performing',
        'outputs/test_results/worst_performing',
        'outputs/test_results/metrics'
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    model.eval()
    
    results = []
    predictions = []
    ground_truths = []
    
    print(f"Testing {len(img_paths)} images...")
    
    for i, img_path in enumerate(img_paths):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
        
        # Load image and ground truth using same method as training
        original_img = Image.open(img_path).convert('RGB')
        img, target = load_data(img_path, train=False)
        
        # Load original count from .mat file
        img_name = os.path.basename(img_path).replace('.jpg', '')
        gt_path = img_path.replace('images', 'ground_truth')
        mat_path = os.path.join(os.path.dirname(gt_path), f'GT_{img_name}.mat')
        original_count = load_mat_annotations(mat_path)
        
        # Transform image for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict density map
        pred_density, pred_count = predict_density_map(model, img_tensor, device)
        
        # Calculate ground truth count (processed)
        gt_count = target.sum()
        
        # Calculate MAE
        mae = abs(pred_count - gt_count)
        
        # Store results
        result = {
            'image_name': os.path.basename(img_path),
            'image_path': img_path,
            'predicted_count': pred_count,
            'ground_truth_count': gt_count,
            'original_count': original_count,
            'mae': mae
        }
        results.append(result)
        
        predictions.append(pred_count)
        ground_truths.append(gt_count)
        
        # Save individual density maps
        img_name_base = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save ground truth density map
        gt_save_path = f'outputs/test_results/density_maps/ground_truth/{img_name_base}_gt.png'
        save_density_map(target, gt_save_path, f'Ground Truth Density (Count: {gt_count:.1f})')
        
        # Save predicted density map
        pred_save_path = f'outputs/test_results/density_maps/predicted/{img_name_base}_pred.png'
        save_density_map(pred_density, pred_save_path, f'Predicted Density (Count: {pred_count:.1f})')
        
        # Save combined visualization
        combined_save_path = f'outputs/test_results/density_maps/combined/{img_name_base}_combined.png'
        combine_and_save_plots(original_img, target, pred_density, combined_save_path,
                              gt_count, pred_count, original_count, mae)
        
        print(f"  Pred: {pred_count:.3f}, GT: {gt_count:.3f}, Original: {original_count}, MAE: {mae:.3f}")
    
    # Calculate overall metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    
    # Save comprehensive results
    save_metrics_to_excel(results, 'outputs/test_results/metrics/detailed_results.xlsx')
    save_best_worst_images(results, 'outputs/test_results')
    
    # Save summary metrics
    summary = {
        'total_images': len(img_paths),
        'mae': mae,
        'rmse': rmse,
        'best_mae': min([r['mae'] for r in results]),
        'worst_mae': max([r['mae'] for r in results])
    }
    
    with open('outputs/test_results/metrics/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return mae, rmse, results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Test CAN-Net with correct method')
    parser.add_argument('--test_json', default='val.json', help='path to test json file')
    parser.add_argument('--model_path', default='outputs/weights/model_best.pth.tar', help='path to model weights')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = CANNet()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from: {args.model_path}")
    
    # Test model comprehensively
    mae, rmse, results = test_model_comprehensive(model, args.test_json, device)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS (Correct Method)")
    print("="*60)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Tested on {len(results)} images")
    print("\nOutputs saved to:")
    print("- outputs/test_results/density_maps/ground_truth/ (GT density maps)")
    print("- outputs/test_results/density_maps/predicted/ (Predicted density maps)")
    print("- outputs/test_results/density_maps/combined/ (Combined visualizations)")
    print("- outputs/test_results/best_performing/ (Best performing images)")
    print("- outputs/test_results/worst_performing/ (Worst performing images)")
    print("- outputs/test_results/metrics/ (Detailed metrics and Excel files)")
    print("="*60)

if __name__ == '__main__':
    main()
