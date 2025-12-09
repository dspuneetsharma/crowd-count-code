import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import scipy.io as sio
from datetime import datetime
import json
import xlsxwriter
from collections import defaultdict

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *

class ComprehensiveTester:
    def __init__(self, model_path, output_dir='test_results'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.data_root = cfg.DATA.DATA_PATH + '/test_data'
        
        # Create output directory structure
        self.setup_directories()
        
        # Initialize model
        self.setup_model()
        
        # Initialize data structures for results
        self.results = []
        self.image_metrics = []
        
        # Setup transforms
        self.setup_transforms()
        
    def setup_directories(self):
        """Create organized directory structure for results"""
        self.dirs = {
            'main': self.output_dir,
            'density_maps': os.path.join(self.output_dir, 'density_maps'),
            'gt_density_maps': os.path.join(self.output_dir, 'gt_density_maps'),
            'pred_density_maps': os.path.join(self.output_dir, 'pred_density_maps'),
            'combined_images': os.path.join(self.output_dir, 'combined_images'),
            'reports': os.path.join(self.output_dir, 'reports'),
            'excel': os.path.join(self.output_dir, 'excel'),
            'best_images': os.path.join(self.output_dir, 'best_images'),
            'worst_images': os.path.join(self.output_dir, 'worst_images')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
    def setup_model(self):
        """Initialize and load the model"""
        wts = torch.FloatTensor([
            0.07259259, 0.05777778, 0.10148148, 0.10592593, 0.10925926,
            0.11, 0.11037037, 0.11074074, 0.11111111, 0.11074074
        ])
        
        self.net = CrowdCounter(ce_weights=wts)
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.cuda()
        self.net.eval()
        
        print(f"Model loaded from: {self.model_path}")
        
    def setup_transforms(self):
        """Setup image transforms"""
        mean_std = cfg.DATA.MEAN_STD
        self.img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        
        self.restore = standard_transforms.Compose([
            own_transforms.DeNormalize(*mean_std),
            standard_transforms.ToPILImage()
        ])
        
    def generate_roi(self):
        """Generate random ROI for testing"""
        ht_img = cfg.TRAIN.INPUT_SIZE[0]
        wd_img = cfg.TRAIN.INPUT_SIZE[1]
        
        roi = torch.zeros((20, 5))
        for i in range(20):
            ht = 0
            wd = 0
            while (ht < (ht_img/4) or wd < (wd_img/4)):
                xmin = random.randint(0, wd_img-2)
                ymin = random.randint(0, ht_img-2)
                xmax = random.randint(0, wd_img-2)
                ymax = random.randint(0, ht_img-2)
                wd = xmax - xmin
                ht = ymax - ymin
                
            roi[i][0] = int(0)
            roi[i][1] = int(xmin)
            roi[i][2] = int(ymin)
            roi[i][3] = int(xmax)
            roi[i][4] = int(ymax)
            
        # Reshape to [K, 5] format expected by ROI pooling
        roi = roi.cuda()
        return roi
        
    def load_ground_truth(self, filename):
        """Load ground truth density map"""
        filename_no_ext = filename.split('.')[0]
        denname = os.path.join(self.data_root, 'den', filename_no_ext + '.csv')
        
        if os.path.exists(denname):
            den = pd.read_csv(denname, sep=',', header=None).values
            den = den.astype(np.float32, copy=False)
            # Apply DEN_ENLARGE multiplication to match training behavior
            den = den * cfg.DATA.DEN_ENLARGE
            return den
        else:
            print(f"Warning: Ground truth file not found: {denname}")
            return None
            
    def preprocess_image(self, img, den):
        """Preprocess image and density map"""
        wd_1, ht_1 = img.size
        
        # Pad if necessary
        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            pad = np.zeros([ht_1, dif])
            img = np.array(img)
            den = np.array(den)
            img = np.hstack((img, pad))
            img = Image.fromarray(img.astype(np.uint8))
            den = np.hstack((den, pad))
            
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            pad = np.zeros([dif, wd_1])
            img = np.array(img)
            den = np.array(den)
            img = np.vstack((img, pad))
            img = Image.fromarray(img.astype(np.uint8))
            den = np.vstack((den, pad))
            
        # Convert to RGB if grayscale
        if img.mode == 'L':
            img = img.convert('RGB')
        
        img = self.img_transform(img)
        return img, den, (ht_1, wd_1)
        
    def predict(self, img, roi):
        """Make prediction using the model"""
        # Don't multiply by 255 - this matches training behavior
        img = img[None, :, :, :].cuda()
        
        with torch.no_grad():
            pred_map, pred_cls, pred_seg = self.net.test_forward(img, roi)
            
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
        pred_count = np.sum(pred_map)
        
        return pred_map, pred_count
        
    def calculate_metrics(self, pred_count, gt_count):
        """Calculate MAE and MSE for a single image"""
        mae = abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        return mae, mse
        
    def save_density_visualization(self, pred_map, gt_map, filename, pred_count, gt_count, original_size):
        """Save density map visualizations"""
        filename_no_ext = filename.split('.')[0]
        
        # Normalize maps
        pred_map_norm = pred_map / np.max(pred_map + 1e-20)
        gt_map_norm = gt_map / np.max(gt_map + 1e-20)
        
        # Crop to original size
        ht_1, wd_1 = original_size
        pred_map_crop = pred_map_norm[0:ht_1, 0:wd_1]
        gt_map_crop = gt_map_norm[0:ht_1, 0:wd_1]
        
        # Create figure with subplots - Original, Ground Truth, Predicted
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        except Exception as e:
            print(f"Warning: Could not create subplot for {filename}: {e}")
            return
        
        # Original image
        img_path = os.path.join(self.data_root, 'images', filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((wd_1, ht_1))
            axes[0].imshow(img)
            axes[0].set_title(f'Original Image\n{filename}')
        else:
            axes[0].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[0].set_title(f'Original Image\n{filename}')
        axes[0].axis('off')
        
        # Ground truth density
        im1 = axes[1].imshow(gt_map_crop, cmap='jet')
        axes[1].set_title(f'Ground Truth\nCount: {gt_count:.1f}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Predicted density
        im2 = axes[2].imshow(pred_map_crop, cmap='jet')
        axes[2].set_title(f'Predicted\nCount: {pred_count:.1f}')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['combined_images'], f'{filename_no_ext}_comparison.png'),
                   bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        # Save individual density maps in separate folders
        self.save_single_density_map(gt_map_crop, f'{filename_no_ext}_gt_{int(gt_count)}.png', 'gt_density_maps')
        self.save_single_density_map(pred_map_crop, f'{filename_no_ext}_pred_{pred_count:.1f}.png', 'pred_density_maps')
                   
    def save_single_density_map(self, density_map, filename, folder='density_maps'):
        """Save a single density map"""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(density_map, cmap='jet')
        ax.set_title(f'Density Map - {filename}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs[folder], filename),
                   bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
    def test_single_image(self, filename):
        """Test a single image and return metrics"""
        print(f"Processing: {filename}")
        
        # Load ground truth
        gt_map = self.load_ground_truth(filename)
        if gt_map is None:
            return None
            
        gt_count = np.sum(gt_map)
        
        # Load and preprocess image
        img_path = os.path.join(self.data_root, 'images', filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            return None
            
        img = Image.open(img_path)
        img, den, original_size = self.preprocess_image(img, gt_map)
        
        # Generate ROI
        roi = self.generate_roi()
        
        # Make prediction
        pred_map, pred_count = self.predict(img, roi)
        
        # Calculate metrics
        mae, mse = self.calculate_metrics(pred_count, gt_count)
        
        # Save visualizations
        self.save_density_visualization(pred_map, gt_map, filename, pred_count, gt_count, original_size)
        
        # Store results
        result = {
            'image_name': filename,
            'predicted_count': float(pred_count),
            'ground_truth_count': float(gt_count),
            'mae': float(mae),
            'mse': float(mse),
            'difference': float(pred_count - gt_count)
        }
        
        return result
        
    def run_comprehensive_test(self):
        """Run comprehensive test on all test images"""
        print("Starting comprehensive testing...")
        
        # Get list of test images
        images_dir = os.path.join(self.data_root, 'images')
        if not os.path.exists(images_dir):
            print(f"Error: Test images directory not found: {images_dir}")
            return
            
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} test images")
        
        # Process each image
        for i, filename in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
            result = self.test_single_image(filename)
            if result:
                self.results.append(result)
                
        print(f"Completed testing on {len(self.results)} images")
        
        # Calculate overall metrics
        self.calculate_overall_metrics()
        
        # Generate reports
        self.generate_reports()
        
    def calculate_overall_metrics(self):
        """Calculate overall MAE and MSE"""
        if not self.results:
            print("No results to calculate metrics from")
            return
            
        mae_values = [r['mae'] for r in self.results]
        mse_values = [r['mse'] for r in self.results]
        
        self.overall_metrics = {
            'total_images': len(self.results),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'mae_min': np.min(mae_values),
            'mae_max': np.max(mae_values),
            'mse_min': np.min(mse_values),
            'mse_max': np.max(mse_values)
        }
        
        print(f"\nOverall Metrics:")
        print(f"Total Images: {self.overall_metrics['total_images']}")
        print(f"MAE: {self.overall_metrics['mae_mean']:.4f} ± {self.overall_metrics['mae_std']:.4f}")
        print(f"MSE: {self.overall_metrics['mse_mean']:.4f} ± {self.overall_metrics['mse_std']:.4f}")
        
    def generate_reports(self):
        """Generate text and Excel reports"""
        self.generate_text_report()
        self.generate_excel_report()
        self.save_best_worst_images()
        
    def generate_text_report(self):
        """Generate detailed text report"""
        report_path = os.path.join(self.dirs['reports'], 'test_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("PCC-Net Comprehensive Test Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: {self.data_root}\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Images: {self.overall_metrics['total_images']}\n")
            f.write(f"MAE (Mean): {self.overall_metrics['mae_mean']:.4f}\n")
            f.write(f"MAE (Std): {self.overall_metrics['mae_std']:.4f}\n")
            f.write(f"MAE (Min): {self.overall_metrics['mae_min']:.4f}\n")
            f.write(f"MAE (Max): {self.overall_metrics['mae_max']:.4f}\n")
            f.write(f"MSE (Mean): {self.overall_metrics['mse_mean']:.4f}\n")
            f.write(f"MSE (Std): {self.overall_metrics['mse_std']:.4f}\n")
            f.write(f"MSE (Min): {self.overall_metrics['mse_min']:.4f}\n")
            f.write(f"MSE (Max): {self.overall_metrics['mse_max']:.4f}\n\n")
            
            # Best performing images (lowest MAE)
            f.write("TOP 10 BEST PERFORMING IMAGES (Lowest MAE)\n")
            f.write("-" * 50 + "\n")
            sorted_by_mae = sorted(self.results, key=lambda x: x['mae'])
            for i, result in enumerate(sorted_by_mae[:10]):
                f.write(f"{i+1:2d}. {result['image_name']:<30} "
                       f"MAE: {result['mae']:8.4f} "
                       f"Pred: {result['predicted_count']:8.1f} "
                       f"GT: {result['ground_truth_count']:8.1f}\n")
            
            f.write("\n")
            
            # Worst performing images (highest MAE)
            f.write("TOP 10 WORST PERFORMING IMAGES (Highest MAE)\n")
            f.write("-" * 50 + "\n")
            for i, result in enumerate(sorted_by_mae[-10:]):
                f.write(f"{i+1:2d}. {result['image_name']:<30} "
                       f"MAE: {result['mae']:8.4f} "
                       f"Pred: {result['predicted_count']:8.1f} "
                       f"GT: {result['ground_truth_count']:8.1f}\n")
            
            f.write("\n")
            
            # Detailed results for all images
            f.write("DETAILED RESULTS FOR ALL IMAGES\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Image Name':<25} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12} {'MAE':<10} {'MSE':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in sorted(self.results, key=lambda x: x['image_name']):
                f.write(f"{result['image_name']:<25} "
                       f"{result['predicted_count']:<12.1f} "
                       f"{result['ground_truth_count']:<12.1f} "
                       f"{result['difference']:<12.1f} "
                       f"{result['mae']:<10.4f} "
                       f"{result['mse']:<10.4f}\n")
                       
        print(f"Text report saved to: {report_path}")
        
    def generate_excel_report(self):
        """Generate Excel report with multiple sheets"""
        excel_path = os.path.join(self.dirs['excel'], 'test_results.xlsx')
        
        workbook = xlsxwriter.Workbook(excel_path)
        
        # Summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.write('A1', 'PCC-Net Test Results Summary')
        summary_sheet.write('A2', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        summary_sheet.write('A3', f'Model: {self.model_path}')
        summary_sheet.write('A4', f'Test Data: {self.data_root}')
        
        # Overall metrics
        summary_sheet.write('A6', 'Overall Metrics')
        summary_sheet.write('A7', 'Total Images')
        summary_sheet.write('B7', self.overall_metrics['total_images'])
        summary_sheet.write('A8', 'MAE (Mean)')
        summary_sheet.write('B8', self.overall_metrics['mae_mean'])
        summary_sheet.write('A9', 'MAE (Std)')
        summary_sheet.write('B9', self.overall_metrics['mae_std'])
        summary_sheet.write('A10', 'MSE (Mean)')
        summary_sheet.write('B10', self.overall_metrics['mse_mean'])
        summary_sheet.write('A11', 'MSE (Std)')
        summary_sheet.write('B11', self.overall_metrics['mse_std'])
        
        # Detailed results sheet
        results_sheet = workbook.add_worksheet('Detailed Results')
        headers = ['Image Name', 'Predicted', 'Ground Truth', 'Difference', 'MAE', 'MSE']
        
        for col, header in enumerate(headers):
            results_sheet.write(0, col, header)
            
        for row, result in enumerate(self.results, 1):
            results_sheet.write(row, 0, result['image_name'])
            results_sheet.write(row, 1, result['predicted_count'])
            results_sheet.write(row, 2, result['ground_truth_count'])
            results_sheet.write(row, 3, result['difference'])
            results_sheet.write(row, 4, result['mae'])
            results_sheet.write(row, 5, result['mse'])
            
        # Best performing images sheet
        best_sheet = workbook.add_worksheet('Best Performing')
        best_sheet.write('A1', 'Top 10 Best Performing Images (Lowest MAE)')
        best_sheet.write('A2', 'Rank')
        best_sheet.write('B2', 'Image Name')
        best_sheet.write('C2', 'Predicted')
        best_sheet.write('D2', 'Ground Truth')
        best_sheet.write('E2', 'Difference')
        best_sheet.write('F2', 'MAE')
        
        sorted_by_mae = sorted(self.results, key=lambda x: x['mae'])
        for i, result in enumerate(sorted_by_mae[:10], 1):
            best_sheet.write(i+2, 0, i)
            best_sheet.write(i+2, 1, result['image_name'])
            best_sheet.write(i+2, 2, result['predicted_count'])
            best_sheet.write(i+2, 3, result['ground_truth_count'])
            best_sheet.write(i+2, 4, result['difference'])
            best_sheet.write(i+2, 5, result['mae'])
            
        # Worst performing images sheet
        worst_sheet = workbook.add_worksheet('Worst Performing')
        worst_sheet.write('A1', 'Top 10 Worst Performing Images (Highest MAE)')
        worst_sheet.write('A2', 'Rank')
        worst_sheet.write('B2', 'Image Name')
        worst_sheet.write('C2', 'Predicted')
        worst_sheet.write('D2', 'Ground Truth')
        worst_sheet.write('E2', 'Difference')
        worst_sheet.write('F2', 'MAE')
        
        for i, result in enumerate(sorted_by_mae[-10:], 1):
            worst_sheet.write(i+2, 0, i)
            worst_sheet.write(i+2, 1, result['image_name'])
            worst_sheet.write(i+2, 2, result['predicted_count'])
            worst_sheet.write(i+2, 3, result['ground_truth_count'])
            worst_sheet.write(i+2, 4, result['difference'])
            worst_sheet.write(i+2, 5, result['mae'])
            
        workbook.close()
        print(f"Excel report saved to: {excel_path}")
        
    def save_best_worst_images(self):
        """Save best and worst performing images separately"""
        if not self.results:
            return
            
        # Sort by MAE
        sorted_results = sorted(self.results, key=lambda x: x['mae'])
        
        # Save best 10 images
        print("Saving best performing images...")
        for i, result in enumerate(sorted_results[:10]):
            filename = result['image_name']
            filename_no_ext = filename.split('.')[0]
            
            # Load the comparison image we already created
            comparison_path = os.path.join(self.dirs['combined_images'], f'{filename_no_ext}_comparison.png')
            if os.path.exists(comparison_path):
                import shutil
                best_path = os.path.join(self.dirs['best_images'], f'best_{i+1:02d}_{filename_no_ext}_mae_{result["mae"]:.2f}.png')
                shutil.copy2(comparison_path, best_path)
        
        # Save worst 10 images
        print("Saving worst performing images...")
        for i, result in enumerate(sorted_results[-10:]):
            filename = result['image_name']
            filename_no_ext = filename.split('.')[0]
            
            # Load the comparison image we already created
            comparison_path = os.path.join(self.dirs['combined_images'], f'{filename_no_ext}_comparison.png')
            if os.path.exists(comparison_path):
                import shutil
                worst_path = os.path.join(self.dirs['worst_images'], f'worst_{i+1:02d}_{filename_no_ext}_mae_{result["mae"]:.2f}.png')
                shutil.copy2(comparison_path, worst_path)
        
        print(f"Best images saved to: {self.dirs['best_images']}")
        print(f"Worst images saved to: {self.dirs['worst_images']}")

def find_best_model(exp_dir='./exp'):
    """Find the best model, prioritizing best_model.pth over highest epoch"""
    best_model = None
    best_epoch = -1
    
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None
    
    # Look for model directories
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if os.path.isdir(item_path):
            # First, check for best_model.pth (highest priority)
            best_model_path = os.path.join(item_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                print(f"Found best_model.pth: {best_model_path}")
                return best_model_path
            
            # If no best_model.pth, look for highest epoch number
            for file in os.listdir(item_path):
                if file.startswith('all_ep_') and file.endswith('.pth'):
                    try:
                        # Extract epoch number
                        epoch_str = file.replace('all_ep_', '').replace('.pth', '')
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_model = os.path.join(item_path, file)
                    except ValueError:
                        continue
    
    return best_model

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Find the best model weights
    model_path = find_best_model()
    
    if model_path is None:
        print("Error: No model found. Please check the exp directory.")
        return
    
    # Create tester and run comprehensive test
    tester = ComprehensiveTester(model_path)
    tester.run_comprehensive_test()
    
    print(f"\nTest completed! Results saved to: {tester.output_dir}")
    print(f"Directory structure:")
    for name, path in tester.dirs.items():
        print(f"  {name}: {path}")

if __name__ == '__main__':
    main()
