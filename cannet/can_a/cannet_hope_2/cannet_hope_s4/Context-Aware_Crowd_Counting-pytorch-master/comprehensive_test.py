import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from cannet import CANNet
from my_dataset import CrowdDataset
import cv2
from PIL import Image

def comprehensive_test(img_root, gt_dmap_root, model_param_path, output_dir='test_results'):
    """
    Comprehensive testing with Excel output and visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'best_worst'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predicted_density'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ground_truth_density'), exist_ok=True)
    
    # Load model with GPU optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CANNet()
    checkpoint = torch.load(model_param_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Enable optimizations for faster inference
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        print(f"GPU optimizations enabled: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU")
    
    # Load dataset with GPU optimization
    dataset = CrowdDataset(img_root, gt_dmap_root, gt_downsample=8, phase='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    
    # Results storage
    results = []
    image_names = []
    
    print("Running comprehensive test...")
    
    # Enable mixed precision for faster inference
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    with torch.no_grad():
        for i, (img, gt_dmap) in enumerate(tqdm(dataloader, desc="Testing")):
            # Move to GPU with non-blocking transfer for faster processing
            img = img.to(device, non_blocking=True)
            gt_dmap = gt_dmap.to(device, non_blocking=True)
            
            # Get image name
            img_name = dataset.img_names[i]
            image_names.append(img_name)
            
            # Forward pass with mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_dmap = model(img)
            else:
                pred_dmap = model(img)
            
            # Calculate counts (keep on GPU for faster computation)
            gt_count = gt_dmap.sum().item()
            pred_count = pred_dmap.sum().item()
            
            # Calculate MAE
            mae = abs(pred_count - gt_count)
            
            # Store results
            results.append({
                'image_name': img_name,
                'original_count': gt_count,
                'predicted_count': pred_count,
                'ground_truth_count': gt_count,  # Same as original_count
                'mae': mae
            })
            
            # Save individual density maps (move to CPU only when needed)
            save_density_maps(img_name, pred_dmap, gt_dmap, output_dir)
            
            # Save individual density maps only
            save_individual_density_maps(img_name, pred_dmap, gt_dmap, output_dir)
            
            # Clear GPU cache periodically to prevent memory issues
            if i % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Note: Comparison images are now saved in save_individual_visualizations
    
    # Create DataFrame and save Excel with multiple sheets
    df = pd.DataFrame(results)
    excel_path = os.path.join(output_dir, 'test_results.xlsx')
    
    # Create summary statistics
    summary_data = {
        'Metric': ['Total Images', 'Average MAE', 'Best MAE', 'Worst MAE', 'RMSE'],
        'Value': [
            len(results),
            f"{df['mae'].mean():.3f}",
            f"{df['mae'].min():.3f}",
            f"{df['mae'].max():.3f}",
            f"{np.sqrt((df['predicted_count'] - df['ground_truth_count']).pow(2).mean()):.3f}"
        ]
    }
    
    # Get top 10 best and worst
    df_sorted = df.sort_values('mae')
    best_10 = df_sorted.head(10)
    worst_10 = df_sorted.tail(10)
    
    # Create best performers summary
    best_summary = best_10[['image_name', 'mae', 'ground_truth_count', 'predicted_count']].copy()
    best_summary['rank'] = range(1, 11)
    best_summary = best_summary[['rank', 'image_name', 'mae', 'ground_truth_count', 'predicted_count']]
    
    # Create worst performers summary
    worst_summary = worst_10[['image_name', 'mae', 'ground_truth_count', 'predicted_count']].copy()
    worst_summary['rank'] = range(1, 11)
    worst_summary = worst_summary[['rank', 'image_name', 'mae', 'ground_truth_count', 'predicted_count']]
    
    # Save Excel with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Results', index=False)
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        best_summary.to_excel(writer, sheet_name='Top 10 Best', index=False)
        worst_summary.to_excel(writer, sheet_name='Top 10 Worst', index=False)
    
    print(f"Excel results saved to: {excel_path}")
    print(f"  - Sheet 1: All Results ({len(results)} images)")
    print(f"  - Sheet 2: Summary Statistics")
    print(f"  - Sheet 3: Top 10 Best Performers")
    print(f"  - Sheet 4: Top 10 Worst Performers")
    
    
    # Save best and worst visualizations with GPU optimization
    save_best_worst_visualizations(best_10, worst_10, dataset, model, device, output_dir, scaler)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total images: {len(results)}")
    print(f"Average MAE: {df['mae'].mean():.3f}")
    print(f"Best MAE: {df['mae'].min():.3f}")
    print(f"Worst MAE: {df['mae'].max():.3f}")
    print(f"Results saved to: {output_dir}")
    
    return df

def save_density_maps(img_name, pred_dmap, gt_dmap, output_dir):
    """Save individual density maps as .npy files"""
    base_name = img_name.replace('.jpg', '')
    
    # Convert to numpy
    pred_np = pred_dmap.squeeze().cpu().numpy()
    gt_np = gt_dmap.squeeze().cpu().numpy()
    
    # Save predicted density map
    pred_path = os.path.join(output_dir, 'density_maps', f'{base_name}_predicted.npy')
    np.save(pred_path, pred_np)
    
    # Save ground truth density map
    gt_path = os.path.join(output_dir, 'density_maps', f'{base_name}_ground_truth.npy')
    np.save(gt_path, gt_np)

def save_individual_density_maps(img_name, pred_dmap, gt_dmap, output_dir):
    """Save individual density maps as separate visualizations"""
    base_name = img_name.replace('.jpg', '')
    
    # Convert tensors to numpy
    pred_np = pred_dmap.squeeze().cpu().numpy()
    gt_np = gt_dmap.squeeze().cpu().numpy()
    
    # 1. Save predicted density map
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pred_np, cmap=CM.jet)
    plt.title(f'Predicted Density Map: {img_name}\nPredicted Count: {pred_np.sum():.1f}')
    plt.colorbar(im)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_density', f'{base_name}_predicted.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Save ground truth density map
    plt.figure(figsize=(8, 6))
    im = plt.imshow(gt_np, cmap=CM.jet)
    plt.title(f'Ground Truth Density Map: {img_name}\nGround Truth Count: {gt_np.sum():.1f}')
    plt.colorbar(im)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ground_truth_density', f'{base_name}_ground_truth.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Save combined visualization (original + predicted + ground truth)
    # Get original image for comparison
    img_path = os.path.join('../part_A/test_data/images', img_name)
    if os.path.exists(img_path):
        img_original = plt.imread(img_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_original)
        axes[0].set_title(f'Original Image\nGT Count: {gt_np.sum():.1f}')
        axes[0].axis('off')
        
        # Predicted density
        im1 = axes[1].imshow(pred_np, cmap=CM.jet)
        axes[1].set_title(f'Predicted Density\nPred Count: {pred_np.sum():.1f}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Ground truth density
        im2 = axes[2].imshow(gt_np, cmap=CM.jet)
        axes[2].set_title(f'Ground Truth Density\nGT Count: {gt_np.sum():.1f}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f'{img_name} - MAE: {abs(pred_np.sum() - gt_np.sum()):.1f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparisons', f'{base_name}_combined.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()


def save_best_worst_visualizations(best_10, worst_10, dataset, model, device, output_dir, scaler=None):
    """Save visualizations for best and worst performing images"""
    
    def create_individual_overview(df_subset, title_prefix, folder_name):
        """Create individual overview files for each best/worst performer"""
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            if idx >= 10:
                break
                
            # Get image index
            img_idx = dataset.img_names.index(row['image_name'])
            img, gt_dmap = dataset[img_idx]
            
            # Convert to batch and predict with GPU optimization
            img_batch = img.unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_dmap = model(img_batch)
                else:
                    pred_dmap = model(img_batch)
            
            # Convert to numpy
            pred_np = pred_dmap.squeeze().cpu().numpy()
            gt_np = gt_dmap.squeeze().numpy()
            
            # Load original image
            img_path = os.path.join('../part_A/test_data/images', row['image_name'])
            if os.path.exists(img_path):
                img_original = plt.imread(img_path)
                
                # Create side-by-side comparison (original, predicted, ground truth)
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'{title_prefix} #{idx+1}: {row["image_name"]} - MAE: {row["mae"]:.1f}', fontsize=14)
                
                # Original image
                axes[0].imshow(img_original)
                axes[0].set_title(f'Original Image\nGT Count: {row["ground_truth_count"]:.1f}')
                axes[0].axis('off')
                
                # Predicted density
                im1 = axes[1].imshow(pred_np, cmap=CM.jet)
                axes[1].set_title(f'Predicted Density\nPred Count: {row["predicted_count"]:.1f}')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1])
                
                # Ground truth density
                im2 = axes[2].imshow(gt_np, cmap=CM.jet)
                axes[2].set_title(f'Ground Truth Density\nGT Count: {row["ground_truth_count"]:.1f}')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'best_worst', f'{folder_name}_{idx+1:02d}_{row["image_name"].replace(".jpg", "")}.png'), 
                            dpi=150, bbox_inches='tight')
                plt.close()
    
    def create_grid_overview(df_subset, title_prefix, folder_name):
        """Create grid overview of all 10 best/worst performers"""
        fig, axes = plt.subplots(3, 10, figsize=(30, 9))
        fig.suptitle(f'{title_prefix} 10 Images (by MAE)', fontsize=16)
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            if idx >= 10:
                break
                
            # Get image index
            img_idx = dataset.img_names.index(row['image_name'])
            img, gt_dmap = dataset[img_idx]
            
            # Convert to batch and predict with GPU optimization
            img_batch = img.unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_dmap = model(img_batch)
                else:
                    pred_dmap = model(img_batch)
            
            # Convert to numpy
            pred_np = pred_dmap.squeeze().cpu().numpy()
            gt_np = gt_dmap.squeeze().numpy()
            
            # Load original image
            img_path = os.path.join('../part_A/test_data/images', row['image_name'])
            if os.path.exists(img_path):
                img_original = plt.imread(img_path)
                
                # Row 1: Original images
                axes[0, idx].imshow(img_original)
                axes[0, idx].set_title(f'{row["image_name"]}\nGT: {row["ground_truth_count"]:.1f}', fontsize=8)
                axes[0, idx].axis('off')
                
                # Row 2: Predicted density maps
                im1 = axes[1, idx].imshow(pred_np, cmap=CM.jet)
                axes[1, idx].set_title(f'Pred: {row["predicted_count"]:.1f}\nMAE: {row["mae"]:.1f}', fontsize=8)
                axes[1, idx].axis('off')
                
                # Row 3: Ground truth density maps
                im2 = axes[2, idx].imshow(gt_np, cmap=CM.jet)
                axes[2, idx].set_title(f'GT Density\nCount: {row["ground_truth_count"]:.1f}', fontsize=8)
                axes[2, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_worst', f'{folder_name}_grid.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create individual overview files
    create_individual_overview(best_10, 'Best', 'best')
    create_individual_overview(worst_10, 'Worst', 'worst')
    
    # Create grid overview files
    create_grid_overview(best_10, 'Best', 'best_10')
    create_grid_overview(worst_10, 'Worst', 'worst_10')
    
    # Create detailed individual files for best and worst with GPU optimization
    save_detailed_best_worst(best_10, worst_10, dataset, model, device, output_dir, scaler)

def save_detailed_best_worst(best_10, worst_10, dataset, model, device, output_dir, scaler=None):
    """Save detailed individual visualizations for best and worst performers"""
    
    def save_detailed_set(df_subset, folder_name):
        os.makedirs(os.path.join(output_dir, 'best_worst', folder_name), exist_ok=True)
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            if idx >= 10:
                break
                
            # Get image index
            img_idx = dataset.img_names.index(row['image_name'])
            img, gt_dmap = dataset[img_idx]
            
            # Convert to batch and predict with GPU optimization
            img_batch = img.unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_dmap = model(img_batch)
                else:
                    pred_dmap = model(img_batch)
            
            # Convert to numpy
            img_np = img.squeeze().numpy().transpose(1, 2, 0)
            pred_np = pred_dmap.squeeze().cpu().numpy()
            gt_np = gt_dmap.squeeze().numpy()
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            # Create detailed comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title(f'Original Image\n{row["image_name"]}')
            axes[0, 0].axis('off')
            
            # Predicted density
            im1 = axes[0, 1].imshow(pred_np, cmap=CM.jet)
            axes[0, 1].set_title(f'Predicted Density\nCount: {row["predicted_count"]:.1f}')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Ground truth density
            im2 = axes[0, 2].imshow(gt_np, cmap=CM.jet)
            axes[0, 2].set_title(f'Ground Truth Density\nCount: {row["ground_truth_count"]:.1f}')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Difference map
            diff_np = np.abs(pred_np - gt_np)
            im3 = axes[1, 0].imshow(diff_np, cmap=CM.jet)
            axes[1, 0].set_title(f'Difference Map\nMAE: {row["mae"]:.1f}')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Error distribution
            axes[1, 1].hist(diff_np.flatten(), bins=50, alpha=0.7)
            axes[1, 1].set_title('Error Distribution')
            axes[1, 1].set_xlabel('Absolute Error')
            axes[1, 1].set_ylabel('Frequency')
            
            # Count comparison
            counts = [row["ground_truth_count"], row["predicted_count"]]
            labels = ['Ground Truth', 'Predicted']
            colors = ['blue', 'red']
            axes[1, 2].bar(labels, counts, color=colors, alpha=0.7)
            axes[1, 2].set_title('Count Comparison')
            axes[1, 2].set_ylabel('Count')
            for i, v in enumerate(counts):
                axes[1, 2].text(i, v + 0.1, f'{v:.1f}', ha='center')
            
            plt.suptitle(f'{folder_name.title()} #{idx+1}: {row["image_name"]} - MAE: {row["mae"]:.1f}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'best_worst', folder_name, f'{idx+1:02d}_{row["image_name"].replace(".jpg", "")}.png'), 
                        dpi=150, bbox_inches='tight')
            plt.close()
    
    # Save detailed best performers
    save_detailed_set(best_10, 'best_detailed')
    
    # Save detailed worst performers
    save_detailed_set(worst_10, 'worst_detailed')

if __name__ == "__main__":
    # Configuration
    img_root = '../part_A/test_data/images'
    gt_dmap_root = '../part_A/test_data/ground_truth'
    model_param_path = './checkpoints/best_mae.pth'
    output_dir = 'test_results'
    
    # Run comprehensive test
    results_df = comprehensive_test(img_root, gt_dmap_root, model_param_path, output_dir)
    
    print("\nComprehensive testing completed!")
    print(f"Check the '{output_dir}' folder for all results and visualizations.")
