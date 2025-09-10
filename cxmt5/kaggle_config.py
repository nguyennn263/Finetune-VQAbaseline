import os
import torch

def get_kaggle_config():
    """Configuration optimized for Kaggle environment with GPU"""
    
    # Detect if running on Kaggle
    is_kaggle = os.path.exists('/kaggle')
    
    if is_kaggle:
        # Kaggle paths
        base_path = '/kaggle/input/auto-vivqa'  # Update with your dataset name
        image_dir = f'{base_path}/images/images'
        text_dir = f'{base_path}/text/text'
        output_dir = '/kaggle/working'
    else:
        # Local paths (fallback)
        base_path = '/home/nguyennn263/Documents/Thesis/Fintune/Dataset'
        image_dir = f'{base_path}/images/images'
        text_dir = f'{base_path}/text/text'
        output_dir = './outputs'
    
    # GPU detection and optimization
    has_gpu = torch.cuda.is_available()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_gpu else 0
    
    # Batch size based on GPU memory
    if gpu_memory > 24:  # High-end GPU
        batch_size = 16
    elif gpu_memory > 15:  # Kaggle P100/T4
        batch_size = 8
    elif gpu_memory > 8:   # Mid-range GPU
        batch_size = 4
    else:                  # Low memory or CPU
        batch_size = 2
    
    config = {
        # Model configurations
        'vision_model': 'openai/clip-vit-base-patch32',
        'text_model': 'xlm-roberta-base', 
        'decoder_model': 'google/mt5-small',
        'hidden_dim': 768,
        'max_length': 128,
        'device': 'cuda' if has_gpu else 'cpu',
        
        # Training parameters - optimized for Kaggle
        'batch_size': batch_size,
        'num_epochs': 2,  # Reduced for Kaggle time limits
        'accumulation_steps': 4 // batch_size,  # Effective batch size of 4
        
        # Paths
        'image_dir': image_dir,
        'output_dir': output_dir,
        'text_dir': text_dir,
        'checkpoint_dir': f'{output_dir}/checkpoints',
        
        # Performance optimizations for Kaggle GPU
        'use_mixed_precision': True,  # Enable AMP for faster training
        'pin_memory': True,
        'num_workers': 2,  # Kaggle has limited CPU cores
        'persistent_workers': True,
        
        # Model specific
        'use_vqkd': True,
        'visual_vocab_size': 8192,
        'num_multiway_layers': 6,
        
        # Staged training (reduced for Kaggle)
        'stage1_epochs': 3,  # Freeze encoders
        'stage2_epochs': 5,  # Partial unfreeze
        
        # Learning rates
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'vision_lr': 5e-6,
        
        # Scheduler
        'warmup_ratio': 0.1,
        'scheduler_type': 'linear_decay_with_warmup',
        'weight_decay': 0.01,
        
        # Regularization
        'label_smoothing': 0.1,
        'dropout_rate': 0.2,
        'unfreeze_last_n_layers': 4,
        
        # Data augmentation
        'use_data_augmentation': True,
        
        # Kaggle specific settings
        'save_checkpoints': True,
        'evaluate_every_n_steps': 300,
        'save_every_n_steps': 600,
        'max_train_time_hours': 8.5,  # Leave buffer for Kaggle 9-hour limit
        'early_stopping_patience': 3,
        
        # Memory optimization
        'gradient_checkpointing': True,
        'dataloader_drop_last': True,
        'empty_cache_every_n_steps': 100,
        
        'version': "1.2",
    }
    
    return config

def print_system_info():
    """Print system information for debugging"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1e9:.1f} GB")
    
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 30)
