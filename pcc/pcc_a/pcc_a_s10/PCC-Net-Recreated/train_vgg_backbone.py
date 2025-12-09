# from tensorboardX import SummaryWriter  # Removed tensorboard dependency
import os
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils


from models.CC import CrowdCounter
from config import cfg
from loading_data import loading_data
from misc.utils import *
from misc.timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
# writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)  # Removed tensorboard
log_txt = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'


if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.mkdir(cfg.TRAIN.EXP_PATH)

# Create experiment directory for log file
exp_dir = os.path.join(cfg.TRAIN.EXP_PATH, exp_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    
pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_mae': 1e20, 'mse':1e20,'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

# Lists to store metrics for plotting
epoch_mae = []
epoch_mse = []
epoch_losses = []
epoch_numbers = []

# Early stopping parameters
early_stopping_patience = 20
early_stopping_counter = 0
best_mae_for_early_stopping = float('inf')

# Text file for saving metrics
metrics_file = os.path.join(exp_dir, 'training_metrics.txt')

# Initialize metrics text file with header
with open(metrics_file, 'w') as f:
    f.write('Training Metrics Log\n')
    f.write('='*50 + '\n')
    f.write('Epoch: MAE, MSE, Loss\n')
    f.write('-'*50 + '\n')

train_set, train_loader, val_set, val_loader, restore_transform = loading_data()

_t = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

rand_seed = cfg.TRAIN.SEED    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = CrowdCounter(ce_weights=train_set.wts,modelname='vgg_backbone')
     

    net.train()

    optimizer = optim.Adam([
                            {'params': [param for name, param in net.named_parameters() if 'seg' in name], 'lr': cfg.TRAIN.SEG_LR},
                            {'params': [param for name, param in net.named_parameters() if 'base' in name], 'lr': 1e-5},
                            {'params': [param for name, param in net.named_parameters() if 'seg' not in name and 'base' not in name], 'lr': cfg.TRAIN.LR}
                          ])
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    
    # i_tb = 0  # Removed tensorboard iteration counter
    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        _t['train time'].tic()
        model_path = train(train_loader, net, optimizer, epoch)
        _t['train time'].toc(average=False)
        print( 'train time of one epoch: {:.2f}s'.format(_t['train time'].diff) )
        if epoch%cfg.VAL.FREQ!=0:
            continue
        _t['val time'].tic()
        should_stop = validate(val_loader, model_path, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print( 'val time of one epoch: {:.2f}s'.format(_t['val time'].diff))

        scheduler.step()
        
        # Check for early stopping
        if should_stop:
            print(f'\n🏁 Training stopped early at epoch {epoch + 1}')
            print(f'   Total epochs completed: {len(epoch_numbers)}')
            break


def train(train_loader, net, optimizer, epoch):
    
    for i, data in enumerate(train_loader, 0):
        _t['iter time'].tic()
        img, gt_map, gt_cnt, roi, gt_roi, gt_seg = data

        for i_img in range(cfg.TRAIN.BATCH_SIZE):
            roi[i_img,:,0] = i_img
        roi = roi.view(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.NUM_BOX,5)
        gt_roi = gt_roi.view(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.NUM_BOX,10)

        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        roi = Variable(roi).cuda().float()
        gt_roi = Variable(gt_roi).cuda()
        gt_seg = Variable(gt_seg).cuda()

        optimizer.zero_grad()
        pred_map,pred_cls, pred_seg = net(img, gt_map, roi, gt_roi, gt_seg)

        loss = net.loss
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            
            loss1,loss2,loss3 = net.f_loss()

            # Removed tensorboard logging

            _t['iter time'].toc(average=False)
            print( '[ep %d][it %d][loss %.8f %.8f %.4f %.4f][%.2fs]' % \
                    (epoch + 1, i + 1, loss.item(), loss1.item(), loss2.item(), loss3.item(), _t['iter time'].diff) )
            # pdb.set_trace()
            print( '        [cnt: gt: %.1f pred: %.6f]' % (gt_cnt[0]/cfg.DATA.DEN_ENLARGE, pred_map[0,:,:,:].sum().item()/cfg.DATA.DEN_ENLARGE) )              
    
    snapshot_name = 'all_ep_%d' % (epoch + 1)
    # save model
    to_saved_weight = []

    if len(cfg.TRAIN.GPU_ID)>1:
        to_saved_weight = net.module.state_dict()                
    else:
        to_saved_weight = net.state_dict()
    model_path = os.path.join(cfg.TRAIN.EXP_PATH, exp_name, snapshot_name + '.pth')
    torch.save(to_saved_weight, model_path)

    return model_path

def validate(val_loader, model_path, epoch, restore):
    net = CrowdCounter(ce_weights=train_set.wts)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    print( '='*50 )
    val_loss_mse = []
    val_loss_cls = []
    val_loss_seg = []
    val_loss = []
    mae = 0.0
    mse = 0.0

    for vi, data in enumerate(val_loader, 0):
        img, gt_map, gt_cnt, roi, gt_roi, gt_seg = data
        # pdb.set_trace()
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_seg = Variable(gt_seg).cuda()

            roi = Variable(roi[0]).cuda().float()
            gt_roi = Variable(gt_roi[0]).cuda()

            pred_map,pred_cls,pred_seg = net(img, gt_map, roi, gt_roi, gt_seg)

            loss1,loss2,loss3 = net.f_loss()
            val_loss_mse.append(loss1.item())
            val_loss_cls.append(loss2.item())
            val_loss_seg.append(loss3.item())
            val_loss.append(net.loss.item())

            pred_map = pred_map.data.cpu().numpy()/cfg.DATA.DEN_ENLARGE
            gt_map = gt_map.data.cpu().numpy()/cfg.DATA.DEN_ENLARGE

            pred_seg = pred_seg.cpu().max(1)[1].squeeze_(1).data.numpy()
            gt_seg = gt_seg.data.cpu().numpy()
            gt_count = np.sum(gt_map)
            pred_cnt = np.sum(pred_map)

            mae += abs(gt_count-pred_cnt)
            mse += ((gt_count-pred_cnt)*(gt_count-pred_cnt))

            # x = []
            # if vi==0:
            #     for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, pred_seg, gt_seg)):
            #         if idx>cfg.VIS.VISIBLE_NUM_IMGS:
            #             break
            #         # pdb.set_trace()
            #         pil_input = restore(tensor[0])
            #         pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
            #         pil_output = torch.from_numpy(tensor[1]/(tensor[1].max()+1e-10)).repeat(3,1,1)
                    
            #         pil_gt_seg = torch.from_numpy(tensor[4]).repeat(3,1,1).float()
            #         pil_pred_seg = torch.from_numpy(tensor[3]).repeat(3,1,1).float()
            #         # pdb.set_trace()
                    
            #         x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output, pil_gt_seg, pil_pred_seg])
            #     x = torch.stack(x, 0)
            #     x = vutils.make_grid(x, nrow=5, padding=5)
            #     writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy()*255).astype(np.uint8))

    mae = mae/val_set.get_num_samples()
    mse = np.sqrt(mse/val_set.get_num_samples())

    '''
    loss1 = float(np.mean(np.array(val_loss_mse)))
    loss2 = float(np.mean(np.array(val_loss_cls)))
    loss3 = float(np.mean(np.array(val_loss_seg)))
    loss = float(np.mean(np.array(val_loss)))'''
    loss1 = np.mean(val_loss_mse)
    loss2 = np.mean(val_loss_cls)
    loss3 = np.mean(val_loss_seg)
    loss = np.mean(val_loss)

    # Removed tensorboard logging


    # Store metrics for plotting
    epoch_mae.append(mae)
    epoch_mse.append(mse)
    epoch_losses.append(loss)
    epoch_numbers.append(epoch + 1)
    
    # Save metrics to text file
    with open(metrics_file, 'a') as f:
        f.write(f'Epoch {epoch + 1:3d}: MAE={mae:6.1f}, MSE={mse:6.1f}, Loss={loss:.8f}\n')

    # Early stopping logic
    global early_stopping_counter, best_mae_for_early_stopping
    if mae < best_mae_for_early_stopping:
        best_mae_for_early_stopping = mae
        early_stopping_counter = 0
        print(f'✓ New best MAE: {mae:.1f} (patience reset)')
    else:
        early_stopping_counter += 1
        print(f'⚠ MAE not improved: {mae:.1f} (patience: {early_stopping_counter}/{early_stopping_patience})')

    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = loss
        
        # Save best model weights
        best_model_path = os.path.join(exp_dir, 'best_model.pth')
        if len(cfg.TRAIN.GPU_ID) > 1:
            best_weights = net.module.state_dict()
        else:
            best_weights = net.state_dict()
        torch.save(best_weights, best_model_path)
        train_record['best_model_name'] = best_model_path
        print(f'💾 Best model saved to: {best_model_path}')        

    print( '='*50 )
    print( exp_name )
    print( '    '+ '-'*20 )
    print( '    [mae %.1f mse %.1f], [val loss %.8f %.8f %.4f %.4f]' % (mae, mse, loss, loss1, loss2, loss3) )        
    print( '    '+ '-'*20 )
    # pdb.set_trace()
    print( '[best] [mae %.1f mse %.1f], [loss %.8f], [epoch %d]' % (train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch']) )
    print( '='*50 )
    
    # Check for early stopping
    if early_stopping_counter >= early_stopping_patience:
        print(f'\n🛑 EARLY STOPPING TRIGGERED!')
        print(f'   MAE has not improved for {early_stopping_patience} epochs')
        print(f'   Best MAE: {best_mae_for_early_stopping:.1f} at epoch {train_record["corr_epoch"]}')
        print(f'   Current MAE: {mae:.1f}')
        return True  # Signal to stop training
    
    return False  # Continue training

def plot_training_metrics():
    """Plot MAE, MSE, and Loss curves separately with best values labeled - called only at end of training"""
    if not epoch_mae:  # No validation data yet
        print("No validation data available for plotting.")
        return
        
    print("\n" + "="*60)
    print("GENERATING TRAINING METRICS PLOTS...")
    print("="*60)
    
    # Find best values
    best_mae_idx = epoch_mae.index(min(epoch_mae))
    best_mae_epoch = epoch_numbers[best_mae_idx]
    best_mae_value = epoch_mae[best_mae_idx]
    
    best_mse_idx = epoch_mse.index(min(epoch_mse))
    best_mse_epoch = epoch_numbers[best_mse_idx]
    best_mse_value = epoch_mse[best_mse_idx]
    
    best_loss_idx = epoch_losses.index(min(epoch_losses))
    best_loss_epoch = epoch_numbers[best_loss_idx]
    best_loss_value = epoch_losses[best_loss_idx]
    
    # 1. MAE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, epoch_mae, 'b-', linewidth=2, label='MAE')
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
    mae_plot_path = os.path.join(exp_dir, 'mae_plot.png')
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    print(f'MAE plot saved to: {mae_plot_path}')
    plt.close()
    
    # 2. MSE Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, epoch_mse, 'g-', linewidth=2, label='MSE')
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
    mse_plot_path = os.path.join(exp_dir, 'mse_plot.png')
    plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
    print(f'MSE plot saved to: {mse_plot_path}')
    plt.close()
    
    # 3. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, epoch_losses, 'r-', linewidth=2, label='Validation Loss')
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
    loss_plot_path = os.path.join(exp_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f'Loss plot saved to: {loss_plot_path}')
    plt.close()
    
    # 4. Combined plot (all metrics in one figure)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # MAE subplot
    ax1.plot(epoch_numbers, epoch_mae, 'b-', linewidth=2, label='MAE')
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
    ax2.plot(epoch_numbers, epoch_mse, 'g-', linewidth=2, label='MSE')
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
    ax3.plot(epoch_numbers, epoch_losses, 'r-', linewidth=2, label='Validation Loss')
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
    combined_plot_path = os.path.join(exp_dir, 'combined_metrics.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f'Combined metrics plot saved to: {combined_plot_path}')
    plt.close()
    
    # Print summary statistics
    print(f'\nTRAINING SUMMARY:')
    print(f'Total epochs completed: {len(epoch_numbers)}')
    print(f'Best MAE: {best_mae_value:.1f} (Epoch {best_mae_epoch})')
    print(f'Best MSE: {best_mse_value:.1f} (Epoch {best_mse_epoch})')
    print(f'Best Loss: {best_loss_value:.6f} (Epoch {best_loss_epoch})')
    print(f'Best model saved to: {train_record["best_model_name"]}')
    print(f'Metrics text file saved to: {metrics_file}')
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    finally:
        # Plot metrics at the end of training
        plot_training_metrics()
