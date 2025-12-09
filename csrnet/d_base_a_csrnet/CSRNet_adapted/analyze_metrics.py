import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

def load_metrics_from_file(metrics_file):
    """Load metrics from a saved metrics file"""
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        return df
    else:
        print(f"Metrics file {metrics_file} not found!")
        return None

def plot_metrics_from_file(metrics_file, save_plot=True):
    """Plot metrics from a saved metrics file"""
    df = load_metrics_from_file(metrics_file)
    if df is None:
        return
    
    epochs = df['Epoch'].values
    train_losses = df['Train_Loss'].values
    val_maes = df['Val_MAE'].values
    learning_rates = df['Learning_Rate'].values
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot validation MAE
    ax2.plot(epochs, val_maes, 'r-', linewidth=2, label='Validation MAE')
    ax2.set_title('Validation MAE vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2, label='Learning Rate')
    ax3.set_title('Learning Rate vs Epochs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = metrics_file.replace('.txt', '_analysis.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Analysis plot saved as {plot_filename}')
    
    plt.show()
    
    return fig

def analyze_training_progress(metrics_file):
    """Analyze training progress and provide insights"""
    df = load_metrics_from_file(metrics_file)
    if df is None:
        return
    
    print("=" * 60)
    print("TRAINING PROGRESS ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"Total training epochs: {len(df)}")
    print(f"Final training loss: {df['Train_Loss'].iloc[-1]:.6f}")
    print(f"Best validation MAE: {df['Val_MAE'].min():.6f}")
    print(f"Final validation MAE: {df['Val_MAE'].iloc[-1]:.6f}")
    
    # Find best epoch
    best_epoch = df['Val_MAE'].idxmin() + 1
    print(f"Best epoch: {best_epoch} (MAE: {df['Val_MAE'].min():.6f})")
    
    # Convergence analysis
    if len(df) > 10:
        recent_mae = df['Val_MAE'].tail(10).values
        mae_std = np.std(recent_mae)
        print(f"Recent MAE stability (last 10 epochs): std={mae_std:.6f}")
        
        if mae_std < 0.01:
            print("✓ Training appears to have converged (low MAE variance)")
        else:
            print("⚠ Training may not have fully converged (high MAE variance)")
    
    # Learning rate analysis
    lr_changes = np.diff(df['Learning_Rate'].values)
    lr_change_epochs = np.where(lr_changes != 0)[0] + 1
    if len(lr_change_epochs) > 0:
        print(f"Learning rate changes at epochs: {lr_change_epochs.tolist()}")
    
    print("=" * 60)

def compare_multiple_runs(metrics_files, labels=None):
    """Compare metrics from multiple training runs"""
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_files))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    for i, metrics_file in enumerate(metrics_files):
        df = load_metrics_from_file(metrics_file)
        if df is None:
            continue
            
        epochs = df['Epoch'].values
        train_losses = df['Train_Loss'].values
        val_maes = df['Val_MAE'].values
        
        # Plot training loss
        ax1.plot(epochs, train_losses, linewidth=2, label=f"{labels[i]} - Train Loss")
        
        # Plot validation MAE
        ax2.plot(epochs, val_maes, linewidth=2, label=f"{labels[i]} - Val MAE")
    
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved as training_comparison.png")

def main():
    """Main function to analyze metrics"""
    print("CSRNet Training Metrics Analysis")
    print("=" * 40)
    
    # Find all metrics files
    metrics_files = glob.glob("*metrics.txt")
    
    if not metrics_files:
        print("No metrics files found! Make sure to run training first.")
        return
    
    print(f"Found {len(metrics_files)} metrics file(s):")
    for i, f in enumerate(metrics_files):
        print(f"  {i+1}. {f}")
    
    # Analyze each file
    for metrics_file in metrics_files:
        print(f"\nAnalyzing: {metrics_file}")
        analyze_training_progress(metrics_file)
        plot_metrics_from_file(metrics_file)
    
    # If multiple files, offer comparison
    if len(metrics_files) > 1:
        print(f"\nMultiple runs detected. Would you like to compare them?")
        response = input("Enter 'y' to compare, any other key to skip: ")
        if response.lower() == 'y':
            labels = [f.replace('metrics.txt', '').replace('_', ' ') for f in metrics_files]
            compare_multiple_runs(metrics_files, labels)

if __name__ == "__main__":
    main()
