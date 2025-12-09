# Context-Aware Crowd Counting - Reimplementation

This is a complete reimplementation of the Context-Aware Crowd Counting (CAN-Net) system, maintaining exact same design logic and architecture as the original, with only data paths updated for the `part_B` dataset structure.

## Files Implemented

### Core Scripts
- **`make_dataset.py`** - Converts MAT point annotations to HDF5 density maps
- **`image.py`** - Image loading and data augmentation utilities
- **`dataset.py`** - PyTorch dataset class for training/testing
- **`model.py`** - CAN-Net model architecture (ContextualModule + CANNet)
- **`utils.py`** - Model saving/loading utilities
- **`train.py`** - Training script with validation
- **`test.py`** - Testing script with 4-quadrant inference
- **`create_json.py`** - JSON file generation for dataset paths

## Key Features

### Identical to Original
- ✅ **Exact same algorithm logic** - No changes to core functionality
- ✅ **Same model architecture** - ContextualModule + CANNet
- ✅ **Same training process** - MSE loss, Adam optimizer, data augmentation
- ✅ **Same inference strategy** - 4-quadrant processing for large images
- ✅ **Same evaluation metrics** - MAE and RMSE calculation

### Updated for part_B Dataset
- 🔄 **Data paths** - Updated from `part_B_final` to `part_B`
- 🔄 **Python 3 compatibility** - Fixed print statements
- 🔄 **Progress display** - Enhanced with file counts and percentages

## Dataset Structure Expected

```
part_B/
├── train_data/
│   ├── images/          # 400 JPG files
│   └── ground_truth/    # 400 MAT + 400 H5 files
└── test_data/
    ├── images/          # 316 JPG files
    └── ground_truth/    # 316 MAT + 316 H5 files
```

## Usage

### 1. Generate Density Maps
```bash
python make_dataset.py
```

### 2. Create JSON Files
```bash
# Update paths in create_json.py first
python create_json.py
```

### 3. Train Model
```bash
python train.py train.json val.json
```

### 4. Test Model
```bash
python test.py
```

## Dependencies

- PyTorch 0.4.1+
- Python 2.7/3.x
- OpenCV
- NumPy
- SciPy
- Matplotlib
- PIL/Pillow
- h5py
- scikit-learn

## Notes

- All scripts maintain exact same functionality as original
- Only data paths updated for `part_B` folder structure
- Enhanced progress display for better user experience
- Python 3 compatible print statements
