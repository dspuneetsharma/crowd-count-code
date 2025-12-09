import h5py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
from model import CANNet
import cv2

def load_data(img_path, train=True):
    """Load and process image and ground truth exactly like in training"""
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])
    
    # Apply the same processing as in training (no augmentation for testing)
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    
    return img, target

def test_model(model, test_json, device):
    """Test model using the same method as training validation"""
    
    # Load test data paths
    with open(test_json, 'r') as f:
        img_paths = json.load(f)
    
    # Transform for images (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    
    predictions = []
    ground_truths = []
    
    print(f"Testing {len(img_paths)} images...")
    
    for i, img_path in enumerate(img_paths):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
        
        # Load image and ground truth using same method as training
        img, target = load_data(img_path, train=False)
        
        # Transform image
        img = transform(img).unsqueeze(0).to(device)
        
        # 4-quadrant inference (same as training validation)
        h, w = img.shape[2:4]
        h_d = h // 2
        w_d = w // 2
        
        img_1 = img[:, :, :h_d, :w_d]
        img_2 = img[:, :, :h_d, w_d:]
        img_3 = img[:, :, h_d:, :w_d]
        img_4 = img[:, :, h_d:, w_d:]
        
        with torch.no_grad():
            density_1 = model(img_1).data.cpu().numpy()
            density_2 = model(img_2).data.cpu().numpy()
            density_3 = model(img_3).data.cpu().numpy()
            density_4 = model(img_4).data.cpu().numpy()
        
        # Calculate counts (same as training validation)
        pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
        target_sum = target.sum()
        
        predictions.append(pred_sum)
        ground_truths.append(target_sum)
        
        print(f"  Pred: {pred_sum:.3f}, GT: {target_sum:.3f}, Error: {abs(pred_sum-target_sum):.3f}")
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    
    return mae, rmse, predictions, ground_truths

def main():
    parser = argparse.ArgumentParser(description='Test CAN-Net with correct method')
    parser.add_argument('--test_json', default='val.json', help='path to test json file')
    parser.add_argument('--model_path', default='outputs/weights/model_best.pth.tar', help='path to model weights')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = CANNet()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from: {args.model_path}")
    
    # Test model
    mae, rmse, predictions, ground_truths = test_model(model, args.test_json, device)
    
    print("\n" + "="*50)
    print("TEST RESULTS (Correct Method)")
    print("="*50)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Tested on {len(predictions)} images")
    print("="*50)
    
    # Save results
    os.makedirs('outputs/test_results', exist_ok=True)
    
    # Save detailed results
    results = {
        'image_paths': [os.path.basename(path) for path in json.load(open(args.test_json))],
        'predictions': predictions.tolist(),
        'ground_truths': ground_truths.tolist(),
        'errors': np.abs(predictions - ground_truths).tolist(),
        'mae': mae,
        'rmse': rmse
    }
    
    with open('outputs/test_results/test_results_correct.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: outputs/test_results/test_results_correct.json")

if __name__ == '__main__':
    main()
