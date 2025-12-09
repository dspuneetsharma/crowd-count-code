#!/usr/bin/env python3
"""
Simple runner script for comprehensive testing
"""

import os
import sys
import argparse
from comprehensive_test import ComprehensiveTester

def find_best_model(exp_dir='./exp'):
    """Find the best model (best_model.pth) or fallback to highest epoch"""
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None
    
    # Look for model directories
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if os.path.isdir(item_path):
            # First try to find best_model.pth
            best_model_path = os.path.join(item_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                return best_model_path
            
            # Fallback to highest epoch model
            best_epoch = -1
            best_model = None
            for file in os.listdir(item_path):
                if file.startswith('all_ep_') and file.endswith('.pth'):
                    try:
                        epoch_str = file.replace('all_ep_', '').replace('.pth', '')
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_model = os.path.join(item_path, file)
                    except ValueError:
                        continue
            if best_model:
                return best_model
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive PCC-Net testing')
    parser.add_argument('--model', type=str, help='Path to model file (if not provided, will find best model)')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Experiment directory to search for models')
    
    args = parser.parse_args()
    
    # Find model if not provided
    if args.model is None:
        print("No model specified, searching for best model...")
        args.model = find_best_model(args.exp_dir)
        
        if args.model is None:
            print("Error: No model found. Please specify a model path with --model")
            return 1
        else:
            print(f"Using best model: {args.model}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Create tester and run test
    try:
        tester = ComprehensiveTester(args.model, args.output)
        tester.run_comprehensive_test()
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results saved to: {tester.output_dir}")
        print(f"\n📊 Directory structure:")
        for name, path in tester.dirs.items():
            print(f"  📂 {name}: {path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
