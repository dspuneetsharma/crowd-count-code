import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    # For training data: '../part_B/train_data/images'
    # For validation data: '../part_B/test_data/images'
    img_folder = '../part_B/test_data/images'

    # path to the final json file
    # Examples: 'train.json', 'val.json', 'test.json'
    output_json = 'val.json'

    img_list = []

    for img_path in glob.glob(join(img_folder,'*.jpg')):
        img_list.append(img_path)

    with open(output_json,'w') as f:
        json.dump(img_list,f)
