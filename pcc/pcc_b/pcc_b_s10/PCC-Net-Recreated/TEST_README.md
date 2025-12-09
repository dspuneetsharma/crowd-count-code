# PCC-Net Comprehensive Testing

This directory contains scripts for comprehensive testing of the PCC-Net model on test data, including detailed metrics calculation, visualization generation, and report creation.

## Files

- `comprehensive_test.py` - Main testing script with comprehensive evaluation
- `run_test.py` - Simple runner script for easy execution
- `requirements_test.txt` - Additional dependencies needed for testing
- `TEST_README.md` - This documentation file

## Features

### Comprehensive Evaluation
- **MAE and MSE calculation** for each image and overall statistics
- **Per-image metrics** with detailed breakdown
- **Top and worst performing images** analysis
- **Ground truth vs predicted count** comparison

### Visualizations
- **Comparison images** showing original, ground truth, and predicted density maps
- **Individual density maps** for each image
- **High-quality visualizations** saved as PNG files
- **MATLAB .mat files** for further analysis

### Reports
- **Text report** with detailed statistics and rankings
- **Excel report** with multiple sheets:
  - Summary with overall metrics
  - Detailed results for all images
  - Best performing images (lowest MAE)
  - Worst performing images (highest MAE)

### Organized Output Structure
```
test_results/
├── images/              # Original images (if needed)
├── combined_images/     # Side-by-side comparison images
│   └── *_comparison.png # Original + GT + Predicted combined
├── gt_density_maps/     # Ground truth density maps only
│   ├── *_gt_*.png      # Ground truth density maps
│   └── *_gt_*.mat      # Ground truth .mat files
├── pred_density_maps/   # Predicted density maps only
│   ├── *_pred_*.png    # Predicted density maps
│   └── *_pred_*.mat    # Predicted .mat files
├── density_maps/        # Legacy folder (if needed)
├── reports/             # Text reports
│   └── test_report.txt  # Detailed text report
├── excel/               # Excel reports
│   └── test_results.xlsx # Comprehensive Excel report
├── best_images/         # Best performing images
└── worst_images/        # Worst performing images
```

## Usage

### Prerequisites

1. Install additional dependencies:
```bash
pip install -r requirements_test.txt
```

2. Ensure your model is trained and saved in the `./exp/` directory

### Running the Test

#### Option 1: Automatic (Recommended)
```bash
python run_test.py
```
This will automatically find the best model (highest epoch) and run the test.

#### Option 2: Specify Model
```bash
python run_test.py --model ./exp/PCC_Net09-12_20-29/all_ep_107.pth
```

#### Option 3: Custom Output Directory
```bash
python run_test.py --output my_test_results
```

#### Option 4: Direct Script Execution
```bash
python comprehensive_test.py
```

### Command Line Options

- `--model`: Path to specific model file
- `--output`: Output directory for results (default: test_results)
- `--exp-dir`: Directory to search for models (default: ./exp)

## Output

### Text Report (`reports/test_report.txt`)
- Overall metrics (MAE, MSE, statistics)
- Top 10 best performing images
- Top 10 worst performing images
- Detailed results for all images

### Excel Report (`excel/test_results.xlsx`)
- **Summary sheet**: Overall metrics and statistics
- **Detailed Results sheet**: All images with individual metrics
- **Best Performing sheet**: Top 10 images with lowest MAE
- **Worst Performing sheet**: Top 10 images with highest MAE

### Visualizations
- **`combined_images/`**: Side-by-side comparison images showing original + ground truth + predicted
- **`gt_density_maps/`**: Ground truth density maps only
- **`pred_density_maps/`**: Predicted density maps only
- **`.mat` files**: MATLAB files for further analysis in respective folders

## Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and ground truth counts
- **MSE (Mean Squared Error)**: Average squared difference between predicted and ground truth counts
- **Difference**: Predicted count - Ground truth count (can be positive or negative)

## Model Selection

The script automatically selects the "best" model based on the highest epoch number found in the experiment directory. If you want to use a specific model, use the `--model` parameter.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model not found**: Check the model path and ensure the file exists
3. **Missing dependencies**: Install requirements with `pip install -r requirements_test.txt`
4. **Test data not found**: Ensure the test data is in the correct directory structure

### Error Messages

- `Model file not found`: Check the model path
- `Test images directory not found`: Verify the data path in config.py
- `Ground truth file not found`: Check if density maps are in the correct location

## Example Output

```
Starting comprehensive testing...
Found 316 test images
Processing 1/316: IMG_001.jpg
Processing 2/316: IMG_002.jpg
...
Completed testing on 316 images

Overall Metrics:
Total Images: 316
MAE: 12.3456 ± 8.9012
MSE: 234.5678 ± 456.7890

Test completed! Results saved to: test_results
```

## Notes

- The script uses the same preprocessing and ROI generation as the original test.py
- All visualizations are saved in high resolution (150 DPI)
- The script maintains the same random seed for reproducibility
- Results are automatically organized in a clean directory structure
