import  h5py
import  scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter as gaussian_filter_old
from matplotlib import cm as CM
import scipy

def gaussian_filter_density(img, points):
    '''
    This code uses k-nearest neighbors to generate a density map with adaptive Gaussian kernels.
    
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    '''
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(img.shape)) / 2. / 2.  # case: 1 point
        density += gaussian_filter_old(pt2d, sigma, mode='constant')
    print('done.')
    return density


# root is the path to ShanghaiTech dataset
root='..'

part_B_train = os.path.join(root,'part_B/train_data','images')
part_B_test = os.path.join(root,'part_B/test_data','images')
path_sets = [part_B_train,part_B_test]


img_paths  = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

total_files = len(img_paths)
print("Total files to process: %d" % total_files)

for i, img_path in enumerate(img_paths):
    progress = i + 1
    percentage = (progress * 100.0) / total_files
    print("Processing %d/%d files (%.1f%%) - %s" % (progress, total_files, percentage, img_path))
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)
    gt = mat["image_info"][0,0][0,0][0]
    
    # Use k-NN adaptive density generation instead of fixed kernel
    k = gaussian_filter_density(img, gt)
    
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

print("Processing completed! Generated %d HDF5 density map files." % total_files)
