import numpy as np
import time
import torch
import torch.nn as nn
import os
import random
import math
import pandas as pd
import argparse
from tqdm import tqdm as tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from cannet import CANNet
from my_dataset import CrowdDataset

def save_ckpt(path, epoch, model, optimizer, scheduler, scaler, best_mae=None, best_epoch=None):
    import os, torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_mae": None if best_mae is None else float(best_mae),
        "best_epoch": None if best_epoch is None else int(best_epoch),
    }
    torch.save(state, path)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.benchmark = True   # variable input sizes -> faster

if __name__=="__main__":
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from (e.g., checkpoints/last.pth)")
    parser.add_argument("--resume_best", action="store_true", help="Also restore running best metrics if present")
    args = parser.parse_args()
    # configuration
    train_image_root='../part_B/train_data/images'
    train_dmap_root='../part_B/train_data/ground_truth'
    test_image_root='../part_B/test_data/images'
    test_dmap_root='../part_B/test_data/ground_truth'
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
    
    # Resume training logic
    start_epoch = 0
    best_mae, best_epoch = float("inf"), -1

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Resuming from: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)

            # restore model
            (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])

            # restore optimizer / scaler / scheduler
            if ckpt.get("optimizer") and optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
                # ensure optimizer state tensors are on the right device
                for s in optimizer.state.values():
                    for k, v in list(s.items()):
                        if torch.is_tensor(v):
                            s[k] = v.to(device)

            if ckpt.get("scaler") and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])

            if ckpt.get("scheduler") and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])

            # epoch + LR schedule continuity
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            if scheduler is not None:
                scheduler.last_epoch = start_epoch - 1

            # optionally restore running best (only if you asked for it)
            if args.resume_best:
                if ckpt.get("best_mae") is not None:  best_mae = float(ckpt["best_mae"])
                if ckpt.get("best_epoch") is not None: best_epoch = int(ckpt["best_epoch"])

            print(f"=> Resume OK | start_epoch={start_epoch} | best_mae={best_mae} | best_epoch={best_epoch}")
        else:
            print(f"=> WARNING: resume path not found: {args.resume}")
    
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
    for epoch in range(start_epoch, max_epochs):
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
        save_ckpt("checkpoints/last.pth", epoch, model, optimizer, scheduler, scaler, best_mae, best_epoch)
        
        # Save best checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            save_ckpt("checkpoints/best_mae.pth", epoch, model, optimizer, scheduler, scaler, best_mae, best_epoch)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    