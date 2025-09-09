#!/usr/bin/env python3
"""
Kaggle Environment Setup Script
This script prepares the environment for running on Kaggle with GPU acceleration
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_kaggle_paths():
    """Setup paths for Kaggle environment"""
    # Kaggle specific paths
    kaggle_input = "/kaggle/input"
    kaggle_working = "/kaggle/working"
    
    # Check if running on Kaggle
    if os.path.exists(kaggle_input):
        print("Detected Kaggle environment")
        return {
            'is_kaggle': True,
            'input_dir': kaggle_input,
            'working_dir': kaggle_working,
            'data_dir': f"{kaggle_input}/your-dataset-name",  # Update this with your dataset name
            'image_dir': f"{kaggle_input}/your-dataset-name/images"  # Update path
        }
    else:
        print("Local environment detected")
        return {
            'is_kaggle': False,
            'input_dir': "./",
            'working_dir': "./",
            'data_dir': "./data",
            'image_dir': "./data/images"
        }

def check_gpu():
    """Check GPU availability"""
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("No GPU available, using CPU")
        return False

def get_kaggle_config():
    """Get configuration optimized for Kaggle"""
    paths = setup_kaggle_paths()
    has_gpu = check_gpu()
    
    config = {
        'vision_model': 'openai/clip-vit-base-patch32',
        'text_model': 'xlm-roberta-base',
        'decoder_model': 'google/mt5-base',
        'hidden_dim': 768,
        'max_length': 128,
        'device': 'cuda' if has_gpu else 'cpu',
        
        # Kaggle GPU optimized settings
        'batch_size': 8 if has_gpu else 2,  # Kaggle GPU has ~16GB
        'num_epochs': 10,
        'accumulation_steps': 2,  # Gradient accumulation for effective larger batch
        
        # Paths
        'image_dir': paths['image_dir'],
        'data_dir': paths['data_dir'],
        'output_dir': paths['working_dir'] + '/outputs',
        
        # Training optimization for Kaggle
        'use_mixed_precision': True,  # Enable for GPU efficiency
        'pin_memory': True,
        'num_workers': 2,  # Kaggle has limited CPU cores
        
        # Model configurations
        'use_vqkd': True,
        'visual_vocab_size': 8192,
        'num_multiway_layers': 6,
        
        # Training strategy
        'stage1_epochs': 3,  # Reduced for Kaggle time limit
        'stage2_epochs': 4,
        
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'vision_lr': 5e-6,
        
        'warmup_ratio': 0.1,
        'scheduler_type': 'linear_decay_with_warmup',
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'dropout_rate': 0.2,
        
        'unfreeze_last_n_layers': 4,
        'use_data_augmentation': True,
        
        # Kaggle specific
        'save_checkpoints': True,
        'evaluate_every_n_steps': 500,
        'save_every_n_steps': 1000,
        'max_train_time_hours': 9,  # Kaggle has 9-hour limit
    }
    
    return config

if __name__ == "__main__":
    print("Setting up Kaggle environment...")
    install_requirements()
    config = get_kaggle_config()
    print("Setup complete!")
    print(f"Configuration: {config}")
