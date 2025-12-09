# Crowd Code Backups

This repository contains crowd counting implementations using three different architectures:

## Directory Structure

### 1. CANNet (Context-Aware Network)
- **can_a/**: CANNet implementations for dataset A
  - `cannet_hope/`: Base implementation
  - `cannet_hope_2/`: Variants with different scales (s4, s10, s15)
- **can_b/**: CANNet implementations for dataset B
  - Various scale implementations (s4, s10, s15)
  - Adaptive implementations

### 2. CSRNet
- **a_s*_csrnet/**: CSRNet implementations for dataset A with different scales
- **b_s*_csrnet/**: CSRNet implementations for dataset B with different scales
- **d_base_*_csrnet/**: Base CSRNet implementations

### 3. PCC (Point Cloud Classification)
- **pcc_a/**: PCC implementations for dataset A
  - Scale variants (s4, s10, s15)
  - Adaptive implementation
- **pcc_b/**: PCC implementations for dataset B
  - Scale variants (s4, s10, s15)
  - Adaptive implementation
- **time/**: Model checkpoints

## Getting Started

Each subdirectory contains its own implementation with training and evaluation scripts. Please refer to the individual README files in each subdirectory for specific instructions.

## Requirements

Python 3.x with PyTorch and other dependencies as specified in individual project requirements.txt files.

