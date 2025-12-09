#!/usr/bin/env python3
"""
Script to regenerate training plots with labeled best values
This script reads existing training metrics and creates new plots with annotations
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def find_latest_experiment(exp_dir='./exp'):
    """Find the most recent experiment directory"""
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None
    
    # Get all experiment directories
    exp_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not exp_dirs:
        print("No experiment directories found")
        return None
    
    # Sort by modification time to get the latest
    exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(exp_dir, x)), reverse=True)
    latest_exp = exp_dirs[0]
    return os.path.join(exp_dir, latest_exp)

def parse_metrics_file(metrics_file):
    """Parse the training metrics text file"""
    epoch_numbers = []
    mae_values = []
    mse_values = []
    loss_values = []
    
    try:
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines:
            if line.startswith('Epoch '):
                # Parse line: "Epoch  66: MAE=  10.6, MSE=  16.6, Loss=0.00001235"
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    epoch_part = parts[0].strip()
                    metrics_part = parts[1].strip()
                    
                    # Extract epoch number
                    epoch_match = re.search(r'Epoch\s+(\d+)', epoch_part)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        epoch_numbers.append(epoch)
                        
                        # Extract MAE, MSE, Loss
                        mae_match = re.search(r'MAE=(\s*\d+\.?\d*)', metrics_part)
                        mse_match = re.search(r'MSE=(\s*\d+\.?\d*)', metrics_part)
                        loss_match = re.search(r'Loss=(\d+\.?\d*)', metrics_part)
                        
                        if mae_match and mse_match and loss_match:
                            mae_values.append(float(mae_match.group(1)))
                            mse_values.append(float(mse_match.group(1)))
                            loss_values.append(float(loss_match.group(1)))
    
    except Exception as e:
        print(f"Error parsing metrics file: {e}")
        return None, None, None, None
    
    return epoch_numbers, mae_values, mse_values, loss_values

def create_plots(exp_dir, epoch_numbers, mae_values, mse_values, loss_values):
    """Create all the plots with annotations"""
    
    if not epoch_numbers:
        print("No data to plot")
        return
    
    # Find best values
    best_mae_idx = mae_values.index(min(mae_values))
    best_mae_epoch = epoch_numbers[best_mae_idx]
    best_mae_value = mae_values[best_mae_idx]
    
    best_mse_idx = mse_values.index(min(mse_values))
    best_mse_epoch = epoch_numbers[best_mse_idx]
    best_mse_value = mse_values[best_mse_idx]
    
    best_loss_idx = loss_values.index(min(loss_values))
    best_loss_epoch = epoch_numbers[best_loss_idx]
    best_loss_value = loss_values[best_loss_idx]
    
    print(f"Best MAE: {best_mae_value:.1f} at epoch {best_mae_epoch}")
    print(f"Best MSE: {best_mse_value:.1f} at epoch {best_mse_epoch}")
    print(f"Best Loss: {best_loss_value:.6f} at epoch {best_loss_epoch}")
    
    # 1. MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, mae_values, 'b-', linewidth=2, label='MAE')
    plt.plot(best_mae_epoch, best_mae_value, 'ro', markersize=8, label=f'Best MAE: {best_mae_value:.1f} (Epoch {best_mae_epoch})')
    plt.annotate(f'Best MAE: {best_mae_value:.1f}', 
                xy=(best_mae_epoch, best_mae_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save MAE plot
    mae_plot_path = os.path.join(exp_dir, 'mae_plot_labeled.png')
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    print(f'MAE plot saved to: {mae_plot_path}')
    plt.close()
    
    # 2. MSE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, mse_values, 'g-', linewidth=2, label='MSE')
    plt.plot(best_mse_epoch, best_mse_value, 'ro', markersize=8, label=f'Best MSE: {best_mse_value:.1f} (Epoch {best_mse_epoch})')
    plt.annotate(f'Best MSE: {best_mse_value:.1f}', 
                xy=(best_mse_epoch, best_mse_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save MSE plot
    mse_plot_path = os.path.join(exp_dir, 'mse_plot_labeled.png')
    plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
    print(f'MSE plot saved to: {mse_plot_path}')
    plt.close()
    
    # 3. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, loss_values, 'r-', linewidth=2, label='Validation Loss')
    plt.plot(best_loss_epoch, best_loss_value, 'ro', markersize=8, label=f'Best Loss: {best_loss_value:.6f} (Epoch {best_loss_epoch})')
    plt.annotate(f'Best Loss: {best_loss_value:.6f}', 
                xy=(best_loss_epoch, best_loss_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save Loss plot
    loss_plot_path = os.path.join(exp_dir, 'loss_plot_labeled.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f'Loss plot saved to: {loss_plot_path}')
    plt.close()
    
    # 4. Combined plot (all metrics in one figure)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # MAE subplot
    ax1.plot(epoch_numbers, mae_values, 'b-', linewidth=2, label='MAE')
    ax1.plot(best_mae_epoch, best_mae_value, 'ro', markersize=8, label=f'Best MAE: {best_mae_value:.1f} (Epoch {best_mae_epoch})')
    ax1.annotate(f'Best MAE: {best_mae_value:.1f}', 
                xy=(best_mae_epoch, best_mae_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error (MAE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE subplot
    ax2.plot(epoch_numbers, mse_values, 'g-', linewidth=2, label='MSE')
    ax2.plot(best_mse_epoch, best_mse_value, 'ro', markersize=8, label=f'Best MSE: {best_mse_value:.1f} (Epoch {best_mse_epoch})')
    ax2.annotate(f'Best MSE: {best_mse_value:.1f}', 
                xy=(best_mse_epoch, best_mse_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss subplot
    ax3.plot(epoch_numbers, loss_values, 'r-', linewidth=2, label='Validation Loss')
    ax3.plot(best_loss_epoch, best_loss_value, 'ro', markersize=8, label=f'Best Loss: {best_loss_value:.6f} (Epoch {best_loss_epoch})')
    ax3.annotate(f'Best Loss: {best_loss_value:.6f}', 
                xy=(best_loss_epoch, best_loss_value), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Validation Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(exp_dir, 'combined_metrics_labeled.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f'Combined metrics plot saved to: {combined_plot_path}')
    plt.close()
    
    # Print summary statistics
    print(f'\nTRAINING SUMMARY:')
    print(f'Total epochs completed: {len(epoch_numbers)}')
    print(f'Best MAE: {best_mae_value:.1f} (Epoch {best_mae_epoch})')
    print(f'Best MSE: {best_mse_value:.1f} (Epoch {best_mse_epoch})')
    print(f'Best Loss: {best_loss_value:.6f} (Epoch {best_loss_epoch})')
    print("="*60)

def main():
    print("="*60)
    print("REGENERATING TRAINING PLOTS WITH LABELED VALUES")
    print("="*60)
    
    # Find the latest experiment directory
    exp_dir = find_latest_experiment()
    if not exp_dir:
        return
    
    print(f"Using experiment directory: {exp_dir}")
    
    # Look for metrics file
    metrics_file = os.path.join(exp_dir, 'training_metrics.txt')
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found: {metrics_file}")
        return
    
    print(f"Reading metrics from: {metrics_file}")
    
    # Parse metrics
    epoch_numbers, mae_values, mse_values, loss_values = parse_metrics_file(metrics_file)
    
    if epoch_numbers is None:
        print("Failed to parse metrics file")
        return
    
    print(f"Found {len(epoch_numbers)} epochs of data")
    
    # Create plots
    create_plots(exp_dir, epoch_numbers, mae_values, mse_values, loss_values)
    
    print("\n✅ All plots regenerated successfully with labeled best values!")
    print("Files created:")
    print(f"  - {os.path.join(exp_dir, 'mae_plot_labeled.png')}")
    print(f"  - {os.path.join(exp_dir, 'mse_plot_labeled.png')}")
    print(f"  - {os.path.join(exp_dir, 'loss_plot_labeled.png')}")
    print(f"  - {os.path.join(exp_dir, 'combined_metrics_labeled.png')}")

if __name__ == '__main__':
    main()
