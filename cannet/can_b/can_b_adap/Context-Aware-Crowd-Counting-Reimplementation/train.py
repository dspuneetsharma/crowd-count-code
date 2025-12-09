import sys
import os

import warnings

from model import CANNet

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

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')

def main():

    global args,best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size    = 8
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 1000
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 4
    
    # Early stopping parameters removed - training will run for full epochs
    
    # Create output directories
    import os
    os.makedirs('outputs/weights', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Metrics tracking
    training_metrics = {
        'epoch': [],
        'mae': [],
        'rmse': [],
        'loss': []
    }
    best_mae = float('inf')
    best_rmse = float('inf')
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    model = CANNet()

    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.decay)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_list, model, criterion, optimizer, epoch)
        mae, rmse = validate(val_list, model, criterion)

        # Track metrics
        training_metrics['epoch'].append(epoch)
        training_metrics['mae'].append(mae)
        training_metrics['rmse'].append(rmse)
        
        # Update best metrics
        if mae < best_mae:
            best_mae = mae
        if rmse < best_rmse:
            best_rmse = rmse

        is_best = mae < best_prec1
        best_prec1 = min(mae, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, is_best, filename='outputs/weights/checkpoint.pth.tar')
        
        # Early stopping logic removed - training will continue for all epochs
    
    # Save metrics to file
    save_metrics_to_file(training_metrics, best_mae, best_rmse)
    
    # Plot metrics
    plot_training_metrics(training_metrics, best_mae, best_rmse)

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
        output = model(img)[:,0,:,:]

        target = target.type(torch.FloatTensor).cuda()
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

def validate(val_list, model, criterion):
    print ('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()

    mae = 0
    rmse = 0

    for i,(img, target) in enumerate(val_loader):
        h,w = img.shape[2:4]
        h_d = h//2
        w_d = w//2
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
        target_sum = target.sum()

        mae += abs(pred_sum-target_sum)
        rmse += (pred_sum-target_sum)**2

    mae = mae/len(val_loader)
    rmse = np.sqrt(rmse/len(val_loader))
    
    print(' * MAE {mae:.3f} * RMSE {rmse:.3f}'
              .format(mae=mae, rmse=rmse))

    return mae, rmse

def save_metrics_to_file(training_metrics, best_mae, best_rmse):
    """Save training metrics to a CSV file"""
    import csv
    
    filename = 'outputs/metrics/training_metrics.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Epoch', 'MAE', 'RMSE'])
        
        # Write data
        for i in range(len(training_metrics['epoch'])):
            writer.writerow([
                training_metrics['epoch'][i],
                training_metrics['mae'][i],
                training_metrics['rmse'][i]
            ])
    
    # Write best metrics summary
    with open('outputs/metrics/best_metrics.txt', 'w') as f:
        f.write(f"Training Summary:\n")
        f.write(f"Best MAE: {best_mae:.6f}\n")
        f.write(f"Best RMSE: {best_rmse:.6f}\n")
        f.write(f"Total Epochs: {len(training_metrics['epoch'])}\n")
    
    print(f"Metrics saved to {filename} and outputs/metrics/best_metrics.txt")

def plot_training_metrics(training_metrics, best_mae, best_rmse):
    """Create and save separate plots for each training metric with best value labels only"""
    
    epochs = training_metrics['epoch']
    mae_values = training_metrics['mae']
    rmse_values = training_metrics['rmse']
    
    # Find the epoch where best values occurred
    best_mae_epoch = epochs[mae_values.index(min(mae_values))]
    best_rmse_epoch = epochs[rmse_values.index(min(rmse_values))]
    
    # Plot 1: MAE
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, mae_values, 'b-', linewidth=2, label='MAE', marker='o', markersize=8)
    plt.scatter([best_mae_epoch], [best_mae], color='red', s=150, zorder=5)
    
    # Add label only for best MAE point
    plt.annotate(f'Best MAE: {best_mae:.3f}\nEpoch {best_mae_epoch}', 
                (best_mae_epoch, best_mae), 
                textcoords="offset points", 
                xytext=(0,20), 
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE) Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/plots/mae_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: RMSE
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, rmse_values, 'm-', linewidth=2, label='RMSE', marker='o', markersize=8)
    plt.scatter([best_rmse_epoch], [best_rmse], color='red', s=150, zorder=5)
    
    # Add label only for best RMSE point
    plt.annotate(f'Best RMSE: {best_rmse:.3f}\nEpoch {best_rmse_epoch}', 
                (best_rmse_epoch, best_rmse), 
                textcoords="offset points", 
                xytext=(0,20), 
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error (RMSE) Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/plots/rmse_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training metrics plots saved in outputs/plots/ folder:")
    print("- mae_plot.png")
    print("- rmse_plot.png")

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
