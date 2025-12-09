import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    
    # Early stopping variables
    patience = 20
    best_mae = float('inf')
    patience_counter = 0
    best_prec1 = float('inf')  # Initialize best_prec1
    
    # Metrics storage for plotting
    train_losses = []
    val_maes = []
    learning_rates = []
    
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training epochs: {args.epochs}")
    print(f"Early stopping patience: 20 epochs")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Task ID: {args.task}")
    print("\nFiles that will be generated:")
    print(f"  - {args.task}checkpoint.pth.tar (latest checkpoint)")
    print(f"  - {args.task}model_best.pth.tar (best model)")
    print(f"  - {args.task}training_metrics.png (training curves)")
    print(f"  - {args.task}metrics.txt (CSV metrics)")
    print("="*60 + "\n")
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        # Train and get training loss
        train_loss = train(train_list, model, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        
        # Validate and get validation MAE
        prec1 = validate(val_list, model, criterion)
        val_maes.append(prec1)
        
        # Store learning rate
        learning_rates.append(args.lr)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        
        # Early stopping check
        if prec1 < best_mae:
            best_mae = prec1
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f' * Early stopping patience: {patience_counter}/{patience}')
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
        
        # Early stopping
        if patience_counter >= patience:
            print(f' * Early stopping triggered after {patience} epochs without improvement')
            break

    # Plot training metrics (always run, even with early stopping)
    print("\n" + "="*60)
    print("GENERATING TRAINING METRICS AND PLOTS")
    print("="*60)

    try:
        # Plot training metrics
        plot_training_metrics(train_losses, val_maes, learning_rates, args.task)
        print("✓ Training metrics plots generated successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
        print("You can still analyze metrics manually using: python analyze_metrics.py")

    try:
        # Save metrics to file
        save_metrics(train_losses, val_maes, learning_rates, args.task)
        print("✓ Training metrics saved to file successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not save metrics: {e}")

    print("="*60)
    print("TRAINING COMPLETED!")
    print(f"Best validation MAE: {best_prec1:.6f}")
    print(f"Total epochs trained: {len(train_losses)}")
    print("="*60)

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        
        
        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
    return losses.avg
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    # Convert CUDA tensor to CPU value for plotting
    if hasattr(mae, 'cpu'):
        mae = mae.cpu().item()
    else:
        mae = mae.item()

    return mae
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

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

def save_metrics(train_losses, val_maes, learning_rates, task_id):
    """Save metrics to a text file for further analysis"""
    with open(f'{task_id}metrics.txt', 'w') as f:
        f.write('Epoch,Train_Loss,Val_MAE,Learning_Rate\n')
        for i in range(len(train_losses)):
            f.write(f'{i+1},{train_losses[i]:.6f},{val_maes[i]:.6f},{learning_rates[i]:.10f}\n')
    print(f'Metrics saved to {task_id}metrics.txt')
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()
