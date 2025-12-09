import numpy as np
import time
import torch
import torch.nn as nn
import os
import random
import math
import pandas as pd
from tqdm import tqdm as tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from cannet import CANNet
from my_dataset import CrowdDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.benchmark = True   # variable input sizes -> faster

if __name__=="__main__":
    # configuration
    train_image_root='../part_A/train_data/images'
    train_dmap_root='../part_A/train_data/ground_truth'
    test_image_root='../part_A/test_data/images'
    test_dmap_root='../part_A/test_data/ground_truth'
    gpu_or_cpu='cuda' # use cuda or cpu
    batch_size        = 1
    max_epochs        = 500
    workers           = 4
    print_freq        = 30
    
    # AMP and gradient accumulation settings
    grad_accum_steps = 4  # effective batch ~4 while keeping BS=1
    max_grad_norm = 1.0 
    
    device=torch.device(gpu_or_cpu)
    model=CANNet().to(device)
    criterion=torch.nn.SmoothL1Loss(reduction='sum')
    
    # AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    
    # Learning rate schedule: warmup → cosine
    base_lr = 1.5e-4
    min_lr = 1e-6
    warmup_epochs = 10

    def lr_lambda(current_epoch):
        # Warmup: linear from 1e-6 to base_lr in first warmup_epochs
        if current_epoch < warmup_epochs:
            start = min_lr
            end = base_lr
            pct = (current_epoch + 1) / warmup_epochs
            return (start + (end - start) * pct) / base_lr
        # Cosine decay from base_lr to min_lr over [warmup_epochs, max_epochs]
        t = (current_epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        target_lr = min_lr + (base_lr - min_lr) * cosine
        return target_lr / base_lr

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.step(0)  # ensure warmup LR is applied at epoch 0
    
    # AMP scaler
    scaler = GradScaler()
    train_dataset=CrowdDataset(train_image_root,train_dmap_root,gt_downsample=8,phase='train')
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,
                                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_dataset=CrowdDataset(test_image_root,test_dmap_root,gt_downsample=8,phase='test')
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,
                                           num_workers=2, pin_memory=True, persistent_workers=True)
    
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    
    # Metrics tracking for Excel
    metrics_data = []
    
    # Best model tracking (no early stopping)
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for step, (img, gt_dmap) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            img = img.to(device, non_blocking=True)
            gt_dmap = gt_dmap.to(device, non_blocking=True)
            
            with autocast():
                et_dmap = model(img)
                loss = criterion(et_dmap, gt_dmap)
                loss_to_scale = loss / grad_accum_steps
            
            scaler.scale(loss_to_scale).backward()
            
            # Track raw training loss
            train_loss_sum += loss.item()
            num_batches += 1
            
            if (step + 1) % grad_accum_steps == 0:
                # Unscale for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        
        # Flush leftover grads if steps not divisible by grad_accum_steps
        if (len(train_loader) % grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate average training loss
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Capture LR used in this epoch (before scheduler.step())
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler after epoch
        scheduler.step()
        
        # Validation phase
        model.eval()
        mae_sum, se_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for img, gt_dmap in tqdm(test_loader, desc="Validation"):
                img = img.to(device, non_blocking=True)
                gt_dmap = gt_dmap.to(device, non_blocking=True)
                
                with autocast():
                    et_dmap = model(img)
                
                # Count by integral (sum of density)
                gt_count = gt_dmap.sum().item()
                pd_count = et_dmap.sum().item()
                mae_sum += abs(pd_count - gt_count)
                se_sum += (pd_count - gt_count) ** 2
                n += 1
        
        val_mae = mae_sum / max(1, n)
        val_rmse = math.sqrt(se_sum / max(1, n))
        
        print(f"[Epoch {epoch:3d}] train_loss={avg_train_loss:.3f} val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f}")
        print(f"    Current best MAE: {best_mae:.3f} (achieved at epoch {best_epoch})")
        
        # Track metrics for Excel
        metrics_data.append({
            'epoch': epoch,
            'learning_rate': current_lr,
            'train_loss': avg_train_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'is_best': val_mae < best_mae,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        })
        
        # Save metrics to Excel after each epoch
        df = pd.DataFrame(metrics_data)
        df.to_excel('./checkpoints/training_metrics.xlsx', index=False, engine='openpyxl')
        if (epoch + 1) % 10 == 0:  # Print save confirmation every 10 epochs to avoid spam
            print(f"    -> Metrics saved to checkpoints/training_metrics.xlsx")
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict()
        }, 'checkpoints/last.pth')
        
        # Save best checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae
            }, 'checkpoints/best_mae.pth')
            print(f"    -> NEW BEST! MAE: {best_mae:.3f} (Epoch {epoch}) - saved checkpoints/best_mae.pth")
        else:
            print(f"    -> No improvement (best MAE: {best_mae:.3f} from epoch {best_epoch})")
        
        # Add spacing after each epoch
        print("-" * 80)
    
    # Final save of all metrics
    df = pd.DataFrame(metrics_data)
    df.to_excel('./checkpoints/training_metrics_final.xlsx', index=False, engine='openpyxl')
    
    # Print training summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total epochs completed: {len(metrics_data)}")
    print(f"Best validation MAE: {best_mae:.3f} (achieved at epoch {best_epoch})")
    print(f"Best validation RMSE: {min([m['val_rmse'] for m in metrics_data]):.3f}")
    print(f"Final learning rate: {metrics_data[-1]['learning_rate']:.6e}")
    print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Metrics saved to: checkpoints/training_metrics_final.xlsx")
    print(f"Best model saved to: checkpoints/best_mae.pth")
    print(f"{'='*80}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    