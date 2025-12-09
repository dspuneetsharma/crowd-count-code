#!/usr/bin/env python3
"""
Test script to verify plotting functionality works correctly
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(train_losses, val_maes, learning_rates, task_id):
    """Plot training metrics and save the figures as separate images"""
    epochs = range(1, len(train_losses) + 1)
    
    # Convert CUDA tensors to CPU numpy arrays if needed
    if hasattr(val_maes[0], 'cpu'):
        val_maes = [mae.cpu().numpy() if hasattr(mae, 'cpu') else mae for mae in val_maes]
    
    # Set matplotlib to non-interactive backend for headless environments
    plt.switch_backend('Agg')
    
    # Plot 1: Training Loss
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=6)
    plt.title('Training Loss vs Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add labels at each point
    for i, (epoch, loss) in enumerate(zip(epochs, train_losses)):
        plt.annotate(f'E{epoch}\n{loss:.2f}', 
                     xy=(epoch, loss), 
                     xytext=(5, 5), 
                     textcoords='offset points',
                     fontsize=10, 
                     ha='left',
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    loss_filename = f'{task_id}training_loss.png'
    plt.savefig(loss_filename, dpi=300, bbox_inches='tight')
    print(f'Training loss plot saved as {loss_filename}')
    plt.close()
    
    # Plot 2: Validation MAE
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_maes, 'r-', linewidth=2, marker='s', markersize=6)
    plt.title('Validation MAE vs Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Validation MAE', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Find and label only the lowest MAE point
    min_mae_idx = np.argmin(val_maes)
    min_epoch = epochs[min_mae_idx]
    min_mae = val_maes[min_mae_idx]
    
    plt.annotate(f'Best: E{min_epoch}\nMAE: {min_mae:.2f}', 
                 xy=(min_epoch, min_mae), 
                 xytext=(10, 10), 
                 textcoords='offset points',
                 fontsize=12, 
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'))
    
    plt.tight_layout()
    mae_filename = f'{task_id}validation_mae.png'
    plt.savefig(mae_filename, dpi=300, bbox_inches='tight')
    print(f'Validation MAE plot saved as {mae_filename}')
    plt.close()
    
    # Plot 3: Learning Rate
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, learning_rates, 'g-', linewidth=2, marker='^', markersize=6)
    plt.title('Learning Rate vs Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add learning rate labels at each point
    for i, (epoch, lr) in enumerate(zip(epochs, learning_rates)):
        plt.annotate(f'E{epoch}\n{lr:.1e}', 
                     xy=(epoch, lr), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     fontsize=10, 
                     ha='center',
                     va='bottom',
                     rotation=45,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    lr_filename = f'{task_id}learning_rate.png'
    plt.savefig(lr_filename, dpi=300, bbox_inches='tight')
    print(f'Learning rate plot saved as {lr_filename}')
    plt.close()
    
    print(f'All three plots saved successfully for task: {task_id}')

# Test with sample data
if __name__ == "__main__":
    # Load metrics from file
    train_losses = []
    val_maes = []
    learning_rates = []
    
    with open('part_Bmetrics.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) == 4:
                train_losses.append(float(parts[1]))
                val_maes.append(float(parts[2]))
                learning_rates.append(float(parts[3]))
    
    print(f"Loaded {len(train_losses)} epochs of data")
    print(f"Training losses: {train_losses[:5]}...")
    print(f"Validation MAEs: {val_maes[:5]}...")
    
    # Test plotting
    plot_training_metrics(train_losses, val_maes, learning_rates, 'part_B')
    print("Plotting test completed!")
