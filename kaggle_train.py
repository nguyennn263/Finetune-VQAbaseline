#!/usr/bin/env python3
"""
Kaggle VQA Training Script
Optimized for Kaggle environment with GPU acceleration
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add current directory to path
sys.path.append('/kaggle/working')
sys.path.append('.')

def setup_environment():
    """Setup Kaggle environment"""
    print("Setting up Kaggle environment...")
    
    # Import after path setup
    from cxmt5.kaggle_config import get_kaggle_config, print_system_info
    
    # Print system info
    print_system_info()
    
    # Get configuration
    config = get_kaggle_config()
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    return config

def main():
    """Main training function for Kaggle"""
    start_time = time.time()
    
    try:
        # Setup environment
        config = setup_environment()
        
        print(f"Training configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Import training modules
        from cxmt5.main import main as train_main
        from cxmt5.config import get_improved_config
        
        # Override config with Kaggle settings
        original_config = get_improved_config()
        original_config.update(config)
        
        # Start training
        print("\n" + "="*50)
        print("Starting VQA Training on Kaggle")
        print("="*50)
        
        # Run training with time monitoring
        train_main()
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/3600:.2f} hours")
        
        # Save final model info
        with open('/kaggle/working/training_info.txt', 'w') as f:
            f.write(f"Training completed successfully\n")
            f.write(f"Total time: {elapsed_time/3600:.2f} hours\n")
            f.write(f"Configuration used:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
        
        print("Training info saved to /kaggle/working/training_info.txt")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        with open('/kaggle/working/error_log.txt', 'w') as f:
            f.write(f"Error occurred: {str(e)}\n")
            f.write(traceback.format_exc())
        
        raise e

if __name__ == "__main__":
    main()
