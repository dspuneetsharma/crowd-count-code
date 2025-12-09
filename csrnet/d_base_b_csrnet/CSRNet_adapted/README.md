# CSRNet Adapted for part_B Dataset

This is the adapted version of CSRNet crowd counting model for your part_B dataset, with Python 3 compatibility and Windows path support.

## Dataset Structure
```
part_B/
├── train_data/
│   ├── images/          (400 JPEG images)
│   └── ground_truth/    (400 MAT files)
└── test_data/
    ├── images/          (316 JPEG images)
    └── ground_truth/    (316 MAT files)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate JSON Files
```bash
python generate_json.py
```
This will create:
- `part_B_train.json` - Training image paths
- `part_B_test.json` - Test image paths  
- `part_B_val.json` - Validation image paths

### 3. Generate HDF5 Density Maps
```bash
python make_dataset.py
```
This converts your .mat ground truth files to HDF5 density maps.

### 4. Train the Model
```bash
python train.py part_B_train.json part_B_val.json 0 0
```

### 5. Validate the Model
```bash
python val.py
```

### 6. Analyze Training Metrics
```bash
python analyze_metrics.py
```

### 7. Test Plotting Functionality (Optional)
```bash
python test_plotting.py
```
This tests that the plotting will work correctly during training.

## Model Architecture
- **Frontend**: VGG16-based feature extractor
- **Backend**: Dilated convolutional layers
- **Output**: Single channel density map
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: SGD with momentum

## Training Parameters
- **Learning Rate**: 1e-7
- **Epochs**: 400 (with early stopping)
- **Batch Size**: 1
- **Momentum**: 0.95
- **Weight Decay**: 5e-4
- **Early Stopping**: Patience of 20 epochs on validation MAE

## New Features

### Early Stopping
- Automatically stops training when validation MAE doesn't improve for 20 epochs
- Prevents overfitting and saves training time
- Saves the best model based on validation performance

### Metrics Tracking & Visualization
- **Per-epoch storage**: Training loss, validation MAE, and learning rate
- **Automatic plotting**: Training curves generated and saved automatically after training
- **Metrics export**: CSV files for further analysis
- **Analysis script**: Comprehensive training progress analysis
- **No manual intervention**: Plots are created automatically when training completes

### Generated Files During Training
- `{task_id}checkpoint.pth.tar` - Latest checkpoint
- `{task_id}model_best.pth.tar` - Best model based on validation MAE
- `{task_id}training_metrics.png` - Training curves visualization (AUTOMATIC)
- `{task_id}metrics.txt` - CSV file with all metrics (AUTOMATIC)

## Files Description
- `model.py` - CSRNet architecture definition
- `train.py` - Training script with early stopping and metrics tracking
- `dataset.py` - Dataset loader
- `image.py` - Image processing utilities
- `utils.py` - Utility functions
- `make_dataset.py` - Ground truth conversion
- `generate_json.py` - JSON file generation
- `val.py` - Validation script
- `make_model.py` - Model testing script
- `analyze_metrics.py` - Training metrics analysis and visualization

## Training Monitoring

### Real-time Output
During training, you'll see:
- Epoch progress and loss updates
- Validation MAE after each epoch
- Early stopping patience counter
- Best MAE tracking

### Post-training Analysis
After training completes:
- Automatic generation of training curves
- Metrics saved to CSV for further analysis
- Use `analyze_metrics.py` for detailed insights

### Metrics Analysis Features
- **Convergence analysis**: Check if training has converged
- **Best epoch identification**: Find the epoch with lowest validation MAE
- **Learning rate analysis**: Track learning rate changes
- **Multiple run comparison**: Compare different training sessions

## Notes
- All images are resized to 1024×768
- Ground truth density maps are generated at 1/8 resolution
- Model expects HDF5 files with 'density' key
- Compatible with Windows paths and Python 3
- Early stopping prevents overfitting and saves training time
- Comprehensive metrics tracking for model analysis
