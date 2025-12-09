import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter  # Fixed import
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from tqdm import tqdm  # Add progress bar

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        # Use fixed sigma value of 4
        sigma = 4
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

# Fixed: Set the root to your correct part_B dataset location
root = '../part_B'  # Relative path from CSRNet_adapted directory

#now generate the part_B's ground truth
part_B_train = os.path.join(root,'train_data','images')
part_B_test = os.path.join(root,'test_data','images')
path_sets = [part_B_train,part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

print(f"Total images to process: {len(img_paths)}")
print("Starting HDF5 density map generation...")

# Process images with progress bar
for img_path in tqdm(img_paths, desc="Generating density maps", unit="image"):
    try:
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0,0][0,0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
                hf['density'] = k
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

print("\nHDF5 density map generation completed!")
print(f"Processed {len(img_paths)} images successfully")

#now see a sample from part_B
if img_paths:
    plt.imshow(Image.open(img_paths[0]))

    gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth,cmap=CM.jet)

    print(np.sum(groundtruth))# don't mind this slight variation
