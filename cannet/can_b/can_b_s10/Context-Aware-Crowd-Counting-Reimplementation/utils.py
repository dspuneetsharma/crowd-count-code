import h5py
import torch
import shutil
import numpy as np

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        # Extract directory and create best model filename
        import os
        dir_name = os.path.dirname(filename)
        if dir_name:
            best_filename = os.path.join(dir_name, 'model_best.pth.tar')
        else:
            best_filename = 'model_best.pth.tar'
        shutil.copyfile(filename, best_filename)
