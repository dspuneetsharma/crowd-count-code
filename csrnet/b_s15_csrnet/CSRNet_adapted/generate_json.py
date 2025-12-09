import os
import glob
import json

# Set the root path to the part_B dataset
root = '../part_B'

# Generate training data paths
train_images_path = os.path.join(root, 'train_data', 'images')
train_images = glob.glob(os.path.join(train_images_path, '*.jpg'))
train_images = [img_path.replace('\\', '/') for img_path in train_images]  # Convert Windows paths to Unix-style

# Generate test data paths
test_images_path = os.path.join(root, 'test_data', 'images')
test_images = glob.glob(os.path.join(test_images_path, '*.jpg'))
test_images = [img_path.replace('\\', '/') for img_path in test_images]  # Convert Windows paths to Unix-style

# Save training JSON
with open('part_B_train.json', 'w') as f:
    json.dump(train_images, f, indent=2)

# Save test JSON
with open('part_B_test.json', 'w') as f:
    json.dump(test_images, f, indent=2)

print(f"Generated JSON files:")
print(f"Training images: {len(train_images)}")
print(f"Test images: {len(test_images)}")
print(f"Files saved: part_B_train.json, part_B_test.json")
