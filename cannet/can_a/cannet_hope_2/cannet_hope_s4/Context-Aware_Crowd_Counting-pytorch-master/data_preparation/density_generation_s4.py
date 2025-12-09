import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
import multiprocessing as mp
from functools import partial


#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points):
    '''
    This code uses fixed sigma value of 4 for all gaussian kernels.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    # Fixed sigma value
    sigma = 4.0
    print(f"Using fixed sigma value: {sigma}")

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        # Use fixed sigma for all points
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


def process_single_image(img_path):
    """
    Process a single image to generate density map
    """
    try:
        print(f"Processing: {img_path}")
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        points = mat["image_info"][0,0][0,0][0]
        k = gaussian_filter_density(img, points)
        # save density_map to disk
        output_path = img_path.replace('.jpg','.npy').replace('images','ground_truth')
        np.save(output_path, k)
        print(f"Completed: {img_path}")
        return f"Success: {img_path}"
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return f"Error: {img_path} - {str(e)}"


# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    root = 'part_A'
    
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root,'train_data','images')
    part_A_test = os.path.join(root,'test_data','images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')
    path_sets = [part_A_train,part_A_test]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    print(f"Total images to process: {len(img_paths)}")
    print(f"Using {12} CPU cores for parallel processing...")
    
    # Use multiprocessing with 12 cores
    with mp.Pool(processes=12) as pool:
        results = pool.map(process_single_image, img_paths)
    
    # Print results summary
    success_count = sum(1 for result in results if result.startswith("Success"))
    error_count = len(results) - success_count
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count} images")
    print(f"Errors: {error_count} images")
    
    if error_count > 0:
        print("\nError details:")
        for result in results:
            if result.startswith("Error"):
                print(result)
    
    '''
    #now see a sample from ShanghaiA
    plt.imshow(Image.open(img_paths[0]))
    
    gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
    plt.imshow(gt_file,cmap=CM.jet)
    
    print(np.sum(gt_file))# don't mind this slight variation
    '''